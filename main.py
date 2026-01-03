import argparse
import torch
import os
import pandas as pd
import numpy as np
import glob
from src.config import (DEVICE, CLASS_NAMES, EXPERIMENTS_DIR, CV_DIR, STATS_DIR, N_FOLDS, 
                        EPOCHS_CUSTOM, EPOCHS_PRETRAINED, LR_CUSTOM, LR_PRETRAINED,
                        PATIENCE_CUSTOM, PATIENCE_PRETRAINED, set_seed)

# Set seed for full reproducibility
set_seed()

from src.data.dataset import (get_data_path, get_dataloaders, get_datasets, 
                               get_sartaj_data_path, get_pkdarabi_data_path, get_external_test_dataset)
from src.models.hybrid import HybridTumorClassifier, TinyHybrid
from src.models.ablation import get_ablation_model, ABLATION_NAMES
from src.models.baselines import get_baselines
from src.training.trainer import train_model
from src.evaluation.metrics import test_and_report
from src.evaluation.interpretability import analyze_with_shap, improved_gradcam_visualization
from src.evaluation.analysis import analyze_model_complexity, analyze_probability_distribution, compare_efficiency
from src.evaluation.performance import measure_model_metrics
from src.visualization.plots import plot_confusion_matrix, create_architecture_diagram
from src.visualization.eda import imshow_samples, plot_class_distribution
from src.evaluation.stats import compare_models

def evaluate_on_test(model, test_loader, name):
    """Evaluate a single model on test set and save confusion matrix."""
    print(f"Evaluating {name} on Test Set...")
    acc, report, auc, cm, per_class = test_and_report(model, test_loader, DEVICE, CLASS_NAMES)
    plot_confusion_matrix(cm, CLASS_NAMES, model_name=name, accuracy=acc, auc=auc)
    print(f"  {name}: Test Acc={acc:.4f}, AUC={auc:.4f}")
    return {'Model': name, 'Test_Accuracy': acc, 'Test_AUC': auc}


def main():
    parser = argparse.ArgumentParser(description="Tumor Classifier Refactored")
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["train", "cv", "ablation", "visualize", "analyze", 
                                 "baselines", "compare", "profile", "eda", "pipeline", "test", "cross_eval"])
    parser.add_argument("--model", type=str, default="Hybrid", help="Model name")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    print(f"Running in {args.mode} mode on {DEVICE}")
    
    # =========================================================================
    # TRAIN MODE (Hybrid only)
    # =========================================================================
    if args.mode == "train":
        epochs = args.epochs if args.epochs else EPOCHS_CUSTOM
        data_path = get_data_path()
        train_ds, val_ds, test_ds = get_datasets(data_path)
        train_loader, val_loader, test_loader = get_dataloaders(train_ds, val_ds, test_ds, batch_size=args.batch_size)
        
        model = HybridTumorClassifier(num_classes=4).to(DEVICE)
        model, stats = train_model(model, train_loader, val_loader, DEVICE, name="Hybrid", epochs=epochs, lr=LR_CUSTOM, patience=PATIENCE_CUSTOM)
        
        # Also evaluate on test
        evaluate_on_test(model, test_loader, "Hybrid")
    
    # =========================================================================
    # CV MODE (Any model)
    # =========================================================================
    elif args.mode == "cv":
        # Determine if this is a Hybrid-based model (uses 1 channel) or pretrained (3 channels)
        is_hybrid_based = args.model in ["Hybrid", "TinyHybrid"] or args.model in ABLATION_NAMES
        channels = 1 if is_hybrid_based else 3
        epochs = args.epochs if args.epochs else (EPOCHS_CUSTOM if is_hybrid_based else EPOCHS_PRETRAINED)
        lr = LR_CUSTOM if is_hybrid_based else LR_PRETRAINED
        patience = PATIENCE_CUSTOM if is_hybrid_based else PATIENCE_PRETRAINED
        
        data_path = get_data_path()
        train_ds_full, _, _ = get_datasets(data_path, channels=channels)
        from sklearn.model_selection import KFold
        from torch.utils.data import Subset
        
        indices = train_ds_full.indices
        base_dataset = train_ds_full.dataset
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        
        fold_results = []
        best_fold_acc = 0.0
        best_fold_weights = None
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
            print(f"\n--- CV Fold {fold+1}/{N_FOLDS} for {args.model} ---")
            train_sub_idx = [indices[i] for i in train_idx]
            val_sub_idx = [indices[i] for i in val_idx]
            
            t_data = Subset(base_dataset, train_sub_idx)
            v_data = Subset(base_dataset, val_sub_idx)
            t_loader, v_loader, _ = get_dataloaders(t_data, v_data, v_data, batch_size=args.batch_size)
            
            if args.model == "Hybrid":
                model = HybridTumorClassifier(num_classes=4).to(DEVICE)
            elif args.model == "TinyHybrid":
                model = TinyHybrid(num_classes=4).to(DEVICE)
            elif args.model in ABLATION_NAMES:
                model = get_ablation_model(args.model, num_classes=4).to(DEVICE)
            else:
                baselines = get_baselines(num_classes=4)
                model = baselines.get(args.model, HybridTumorClassifier(num_classes=4)).to(DEVICE)

            model, stats = train_model(model, t_loader, v_loader, DEVICE, name=f"{args.model}_Fold{fold}", epochs=epochs, lr=lr, patience=patience)
            fold_acc = max(stats['val_acc'])
            fold_results.append(fold_acc)
            
            # Track best fold
            if fold_acc > best_fold_acc:
                best_fold_acc = fold_acc
                best_fold_weights = model.state_dict().copy()
            
        print(f"CV Results ({args.model}): {fold_results}, Mean: {np.mean(fold_results):.4f}")
        safe_name = args.model.lower().replace(" ", "_")
        
        # Save best fold as main model for visualization
        if best_fold_weights is not None:
            torch.save(best_fold_weights, os.path.join(EXPERIMENTS_DIR, f"{safe_name}_model.pth"))
            print(f"Best fold model saved to {EXPERIMENTS_DIR}/{safe_name}_model.pth")
        
        torch.save({args.model: fold_results}, os.path.join(EXPERIMENTS_DIR, f"cv_results_{safe_name}.pkl"))

    # =========================================================================
    # BASELINES MODE
    # =========================================================================
    elif args.mode == "baselines":
        epochs = args.epochs if args.epochs else EPOCHS_PRETRAINED
        data_path = get_data_path()
        train_ds, val_ds, test_ds = get_datasets(data_path, channels=3)
        train_loader, val_loader, test_loader = get_dataloaders(train_ds, val_ds, test_ds, batch_size=args.batch_size)
        
        models_dict = get_baselines(num_classes=4)
        results = []
        for name, model in models_dict.items():
            print(f"\n--- Baseline: {name} ---")
            model = model.to(DEVICE)
            model, stats = train_model(model, train_loader, val_loader, DEVICE, name=name, epochs=epochs, lr=LR_PRETRAINED, patience=PATIENCE_PRETRAINED)
            # Evaluate on test
            res = evaluate_on_test(model, test_loader, name)
            results.append(res)
            
            # Explicit memory cleanup
            del model
            torch.cuda.empty_cache()
        
        pd.DataFrame(results).to_csv(os.path.join(EXPERIMENTS_DIR, "baselines_test_results.csv"), index=False)


    # =========================================================================
    # ABLATION MODE
    # =========================================================================
    elif args.mode == "ablation":
        epochs = args.epochs if args.epochs else EPOCHS_CUSTOM
        data_path = get_data_path()
        train_ds, val_ds, test_ds = get_datasets(data_path)
        train_loader, val_loader, test_loader = get_dataloaders(train_ds, val_ds, test_ds, batch_size=args.batch_size)
        
        results = []
        for name in ABLATION_NAMES:
            print(f"\n--- Ablation: {name} ---")
            model = get_ablation_model(name, num_classes=4).to(DEVICE)
            model, stats = train_model(model, train_loader, val_loader, DEVICE, name=name, epochs=epochs, lr=LR_CUSTOM, patience=PATIENCE_CUSTOM)
            res = evaluate_on_test(model, test_loader, name)
            results.append(res)
        pd.DataFrame(results).to_csv(os.path.join(EXPERIMENTS_DIR, "ablation_test_results.csv"), index=False)

    # =========================================================================
    # TEST MODE (Evaluate all saved models)
    # =========================================================================
    elif args.mode == "test":
        data_path = get_data_path()
        _, _, test_ds_1ch = get_datasets(data_path, channels=1)
        _, _, test_ds_3ch = get_datasets(data_path, channels=3)
        _, _, test_loader_1ch = get_dataloaders(test_ds_1ch, test_ds_1ch, test_ds_1ch, batch_size=args.batch_size)
        _, _, test_loader_3ch = get_dataloaders(test_ds_3ch, test_ds_3ch, test_ds_3ch, batch_size=args.batch_size)
        
        results = []
        # Hybrid
        path = os.path.join(EXPERIMENTS_DIR, "hybrid_model.pth")
        if os.path.exists(path):
            model = HybridTumorClassifier(num_classes=4).to(DEVICE)
            model.load_state_dict(torch.load(path))
            results.append(evaluate_on_test(model, test_loader_1ch, "Hybrid"))
            del model
            torch.cuda.empty_cache()
        
        # TinyHybrid
        path = os.path.join(EXPERIMENTS_DIR, "tinyhybrid_model.pth")
        if os.path.exists(path):
            model = TinyHybrid(num_classes=4).to(DEVICE)
            model.load_state_dict(torch.load(path))
            results.append(evaluate_on_test(model, test_loader_1ch, "TinyHybrid"))
            del model
            torch.cuda.empty_cache()
        
        # Ablations (1 channel)
        for name in ABLATION_NAMES:
            safe_name = name.lower().replace(" ", "_")
            # Prioritize _model.pth (best fold from CV) over .pth (last fold from trainer)
            path = os.path.join(EXPERIMENTS_DIR, f"{safe_name}_model.pth")
            if not os.path.exists(path):
                path = os.path.join(EXPERIMENTS_DIR, f"{safe_name}.pth")
            if os.path.exists(path):
                model = None
                try:
                    model = get_ablation_model(name, num_classes=4).to(DEVICE)
                    model.load_state_dict(torch.load(path))
                    results.append(evaluate_on_test(model, test_loader_1ch, name))
                except RuntimeError as e:
                    print(f"  Skipping {name}: incompatible weights (architecture mismatch)")
                finally:
                    if model is not None:
                        del model
                    torch.cuda.empty_cache()
        
        # Baselines (3 channel)
        baselines = get_baselines(num_classes=4)
        for baseline_name, model_fn in baselines.items():
            safe_name = baseline_name.lower().replace(" ", "_")
            # Prioritize _model.pth (best fold from CV) over .pth (last fold from trainer)
            path = os.path.join(EXPERIMENTS_DIR, f"{safe_name}_model.pth")
            if not os.path.exists(path):
                path = os.path.join(EXPERIMENTS_DIR, f"{safe_name}.pth")
            if os.path.exists(path):
                model = None
                try:
                    model = model_fn.to(DEVICE)
                    model.load_state_dict(torch.load(path))
                    results.append(evaluate_on_test(model, test_loader_3ch, baseline_name))
                except RuntimeError as e:
                    print(f"  Skipping {baseline_name}: incompatible weights")
                finally:
                    if model is not None:
                        del model
                    torch.cuda.empty_cache()
                
        if results:
            pd.DataFrame(results).to_csv(os.path.join(EXPERIMENTS_DIR, "all_test_results.csv"), index=False)
            print("\nAll results saved to experiments/all_test_results.csv")

    # =========================================================================
    # PIPELINE MODE (Full Overnight Run)
    # =========================================================================
    elif args.mode == "pipeline":
        from sklearn.model_selection import KFold
        from torch.utils.data import Subset
        from src.visualization.plots import plot_metrics_per_model, display_predictions_with_probabilities
        
        print("="*60)
        print("STARTING FULL PIPELINE WITH CV (Overnight Run)")
        print("="*60)
        
        # 1. EDA
        print("\n[1/7] Running EDA...")
        data_path = get_data_path()
        train_ds, val_ds, test_ds = get_datasets(data_path)
        train_loader, val_loader, test_loader = get_dataloaders(train_ds, val_ds, test_ds, batch_size=args.batch_size)
        imshow_samples(train_loader, title="Pipeline_Train_Samples")
        plot_class_distribution(train_ds, val_ds, test_ds, CLASS_NAMES)
        
        # 2. CV for Hybrid-based models (Hybrid, TinyHybrid, Ablations)
        print("\n[2/7] Cross-Validation for Hybrid-based models...")
        epochs_hybrid = args.epochs if args.epochs else EPOCHS_CUSTOM
        train_ds_full, _, _ = get_datasets(data_path, channels=1)
        indices = train_ds_full.indices
        base_dataset = train_ds_full.dataset
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        
        # All Hybrid-based models to train with CV
        hybrid_models_config = {
            "Hybrid": lambda: HybridTumorClassifier(num_classes=4),
            "TinyHybrid": lambda: TinyHybrid(num_classes=4),
        }
        # Add ablations
        for abl_name in ABLATION_NAMES:
            hybrid_models_config[abl_name] = lambda n=abl_name: get_ablation_model(n, num_classes=4)
        
        for model_name, model_fn in hybrid_models_config.items():
            print(f"\n  CV for {model_name}...")
            cv_results = []
            best_acc = 0.0
            best_weights = None
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
                t_data = Subset(base_dataset, [indices[i] for i in train_idx])
                v_data = Subset(base_dataset, [indices[i] for i in val_idx])
                t_loader, v_loader, _ = get_dataloaders(t_data, v_data, v_data, batch_size=args.batch_size)
                
                model = model_fn().to(DEVICE)
                model, stats = train_model(model, t_loader, v_loader, DEVICE, 
                                          name=f"{model_name}_Fold{fold}", epochs=epochs_hybrid, lr=LR_CUSTOM, patience=PATIENCE_CUSTOM)
                fold_acc = max(stats['val_acc'])
                cv_results.append(fold_acc)
                
                if fold_acc > best_acc:
                    best_acc = fold_acc
                    best_weights = model.state_dict().copy()
            
            # Save best weights from CV
            safe_name = model_name.lower().replace(" ", "_")
            torch.save(best_weights, os.path.join(EXPERIMENTS_DIR, f"{safe_name}_model.pth"))
            torch.save({model_name: cv_results}, os.path.join(EXPERIMENTS_DIR, f"cv_results_{safe_name}.pkl"))
            print(f"  {model_name} CV: {cv_results}, Mean: {np.mean(cv_results):.4f}")
        
        # 3. CV for Baselines (3-channel pretrained models)
        print("\n[3/7] Cross-Validation for Baselines...")
        train_ds_3ch, _, test_ds_3ch = get_datasets(data_path, channels=3)
        _, _, test_loader_3ch = get_dataloaders(train_ds_3ch, train_ds_3ch, test_ds_3ch, batch_size=args.batch_size)
        
        indices_3ch = train_ds_3ch.indices
        base_dataset_3ch = train_ds_3ch.dataset
        baselines = get_baselines(num_classes=4)
        
        for baseline_name in baselines.keys():
            print(f"\n  CV for {baseline_name}...")
            cv_results = []
            best_acc = 0.0
            best_weights = None
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(indices_3ch)):
                t_data = Subset(base_dataset_3ch, [indices_3ch[i] for i in train_idx])
                v_data = Subset(base_dataset_3ch, [indices_3ch[i] for i in val_idx])
                t_loader, v_loader, _ = get_dataloaders(t_data, v_data, v_data, batch_size=args.batch_size)
                
                model = get_baselines(num_classes=4)[baseline_name].to(DEVICE)
                model, stats = train_model(model, t_loader, v_loader, DEVICE, 
                                          name=f"{baseline_name}_Fold{fold}", epochs=EPOCHS_PRETRAINED, lr=LR_PRETRAINED, patience=PATIENCE_PRETRAINED)
                fold_acc = max(stats['val_acc'])
                cv_results.append(fold_acc)
                
                if fold_acc > best_acc:
                    best_acc = fold_acc
                    best_weights = model.state_dict().copy()
            
            # Save best weights from CV
            safe_name = baseline_name.lower().replace(" ", "_")
            torch.save(best_weights, os.path.join(EXPERIMENTS_DIR, f"{safe_name}_model.pth"))
            torch.save({baseline_name: cv_results}, os.path.join(EXPERIMENTS_DIR, f"cv_results_{safe_name}.pkl"))
            print(f"  {baseline_name} CV: {cv_results}, Mean: {np.mean(cv_results):.4f}")
        
        # 4. Test Evaluation for ALL models
        print("\n[4/7] Final Test Evaluation...")
        from src.visualization.plots import plot_final_results_table
        _, _, test_ds_1ch = get_datasets(data_path, channels=1)
        _, _, test_loader_1ch = get_dataloaders(test_ds_1ch, test_ds_1ch, test_ds_1ch, batch_size=args.batch_size)
        final_results = []
        
        # Test Hybrid-based models (1 channel)
        for model_name in hybrid_models_config.keys():
            safe_name = model_name.lower().replace(" ", "_")
            path = os.path.join(EXPERIMENTS_DIR, f"{safe_name}_model.pth")
            if os.path.exists(path):
                if model_name == "Hybrid":
                    model = HybridTumorClassifier(num_classes=4).to(DEVICE)
                elif model_name == "TinyHybrid":
                    model = TinyHybrid(num_classes=4).to(DEVICE)
                else:
                    model = get_ablation_model(model_name, num_classes=4).to(DEVICE)
                try:
                    model.load_state_dict(torch.load(path))
                    final_results.append(evaluate_on_test(model, test_loader_1ch, model_name))
                except RuntimeError as e:
                    print(f"  Skipping {model_name} - incompatible weights")
        
        # Test Baselines (3 channel)
        for baseline_name in baselines.keys():
            safe_name = baseline_name.lower().replace(" ", "_")
            path = os.path.join(EXPERIMENTS_DIR, f"{safe_name}_model.pth")
            if os.path.exists(path):
                model = get_baselines(num_classes=4)[baseline_name].to(DEVICE)
                model.load_state_dict(torch.load(path))
                final_results.append(evaluate_on_test(model, test_loader_3ch, baseline_name))
        
        # Save results table
        plot_final_results_table(final_results)
        
        # 5. Statistical Comparison
        print("\n[5/7] Statistical Comparison...")
        all_cv_results = {}
        for f in glob.glob(os.path.join(EXPERIMENTS_DIR, "cv_results_*.pkl")):
            all_cv_results.update(torch.load(f))
        if len(all_cv_results) >= 2:
            compare_models(all_cv_results)
        
        # 6. Visualizations
        print("\n[6/7] Generating Visualizations...")
        create_architecture_diagram()
        model = HybridTumorClassifier(num_classes=4).to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(EXPERIMENTS_DIR, "hybrid_model.pth")))
        analyze_probability_distribution(model, test_loader, DEVICE, CLASS_NAMES)
        display_predictions_with_probabilities(model, test_loader, DEVICE, CLASS_NAMES)
        
        # 7. Efficiency Comparison
        print("\n[7/7] Efficiency Comparison...")
        all_models = {name: (m, 3) for name, m in get_baselines(num_classes=4).items()}
        all_models['Hybrid'] = (HybridTumorClassifier(num_classes=4), 1)
        all_models['TinyHybrid'] = (TinyHybrid(num_classes=4), 1)
        compare_efficiency(all_models, device=DEVICE)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)


    # =========================================================================
    # VISUALIZE MODE
    # =========================================================================
    elif args.mode == "visualize":
        from src.evaluation.interpretability import (
            gradcam_captum, integrated_gradients_captum, occlusion_sensitivity,
            visualize_ablation_comparison, hybrid_attention_compact
        )
        
        print("Generating Architecture Diagram...")
        create_architecture_diagram()
        
        data_path = get_data_path()
        
        # Hybrid Model (1 channel) - Use new Hybrid Attention method
        hybrid_path = os.path.join(EXPERIMENTS_DIR, "hybrid_model.pth")
            
        if os.path.exists(hybrid_path):
            print("\n--- Hybrid Model (Hybrid Attention) ---")
            model = HybridTumorClassifier(num_classes=4).to(DEVICE)
            model.load_state_dict(torch.load(hybrid_path, map_location=DEVICE))
            _, _, test_ds = get_datasets(data_path, channels=1)
            _, _, test_loader = get_dataloaders(test_ds, test_ds, test_ds, batch_size=4, shuffle_test=False)
            
            # 4-row analysis (Original, CNN, SmoothGrad, Transformer)
            try:
                from src.evaluation.interpretability import hybrid_attention_map
                hybrid_attention_map(model, test_loader, DEVICE, CLASS_NAMES, model_name="Hybrid")
            except Exception as e:
                print(f"Hybrid 4-row failed: {e}")
            
            # Compact single-row visualization (just hybrid attention)
            try:
                hybrid_attention_compact(model, test_loader, DEVICE, CLASS_NAMES, model_name="Hybrid")
            except Exception as e:
                print(f"Hybrid Compact failed: {e}")
            
            # Also run the other methods for comparison
            try:
                from src.evaluation.interpretability import analyze_with_shap
                analyze_with_shap(model, test_loader, DEVICE, CLASS_NAMES, model_name="Hybrid")
            except Exception as e:
                print(f"Hybrid SHAP failed: {e}")
            try:
                integrated_gradients_captum(model, test_loader, DEVICE, CLASS_NAMES, model_name="Hybrid")
            except Exception as e:
                print(f"Hybrid IG failed: {e}")
            try:
                occlusion_sensitivity(model, test_loader, DEVICE, CLASS_NAMES, model_name="Hybrid")
            except Exception as e:
                print(f"Hybrid Occlusion failed: {e}")
        
        # Ablation Comparison - visualize all ablation models side by side
        print("\n--- Ablation Model Comparison ---")
        _, _, test_ds = get_datasets(data_path, channels=1)
        _, _, test_loader = get_dataloaders(test_ds, test_ds, test_ds, batch_size=4, shuffle_test=False)
        
        # Use EXPERIMENTS_DIR directly (already experiments/models)
        models_dir = EXPERIMENTS_DIR
            
        try:
            visualize_ablation_comparison(test_loader, DEVICE, CLASS_NAMES, models_dir)
        except Exception as e:
            print(f"Ablation comparison failed: {e}")
            import traceback
            traceback.print_exc()
        
        # DenseNet (3 channel) - GradCAM works well for pure CNNs
        densenet_path = os.path.join(EXPERIMENTS_DIR, "densenet121_model.pth")
            
        if os.path.exists(densenet_path):
            print("\n--- DenseNet121 (GradCAM) ---")
            model = get_baselines(num_classes=4)['DenseNet121'].to(DEVICE)
            model.load_state_dict(torch.load(densenet_path, map_location=DEVICE))
            _, _, test_ds_3ch = get_datasets(data_path, channels=3)
            _, _, test_loader_3ch = get_dataloaders(test_ds_3ch, test_ds_3ch, test_ds_3ch, batch_size=4, shuffle_test=False)
            try:
                # Use the last dense block's last conv layer
                target_layer = model.features.denseblock4.denselayer16.conv2
                gradcam_captum(model, test_loader_3ch, DEVICE, CLASS_NAMES, model_name="DenseNet121",
                              target_layer=target_layer)
            except Exception as e:
                print(f"DenseNet GradCAM failed: {e}")
        
        # ViT (3 channel) - Use Integrated Gradients (better for Transformers)
        vit_path = os.path.join(EXPERIMENTS_DIR, "vit_model.pth")
            
        if os.path.exists(vit_path):
            print("\n--- ViT (Integrated Gradients) ---")
            model = get_baselines(num_classes=4)['ViT'].to(DEVICE)
            model.load_state_dict(torch.load(vit_path, map_location=DEVICE))
            _, _, test_ds_3ch = get_datasets(data_path, channels=3)
            _, _, test_loader_3ch = get_dataloaders(test_ds_3ch, test_ds_3ch, test_ds_3ch, batch_size=4, shuffle_test=False)
            try:
                integrated_gradients_captum(model, test_loader_3ch, DEVICE, CLASS_NAMES, model_name="ViT")
            except Exception as e:
                print(f"ViT visualization failed: {e}")

    # =========================================================================
    # ANALYZE MODE
    # =========================================================================
    elif args.mode == "analyze":
        # Probability Analysis (Hybrid)
        model_path = os.path.join(EXPERIMENTS_DIR, "hybrid_model.pth")
        if os.path.exists(model_path):
            model = HybridTumorClassifier(num_classes=4).to(DEVICE)
            model.load_state_dict(torch.load(model_path))
            data_path = get_data_path()
            _, _, test_ds = get_datasets(data_path)
            _, _, test_loader = get_dataloaders(test_ds, test_ds, test_ds, batch_size=32)
            analyze_model_complexity(model, input_res=(1, 224, 224))
            analyze_probability_distribution(model, test_loader, DEVICE, CLASS_NAMES)
        
        # Efficiency Comparison (All models with correct channels)
        print("\nRunning Efficiency Comparison...")
        all_models = {name: (m, 3) for name, m in get_baselines(num_classes=4).items()}
        all_models['Hybrid'] = (HybridTumorClassifier(num_classes=4), 1)
        all_models['TinyHybrid'] = (TinyHybrid(num_classes=4), 1)
        compare_efficiency(all_models, device=DEVICE)
            
    # =========================================================================
    # PROFILE MODE
    # =========================================================================
    elif args.mode == "profile":
        measure_model_metrics(lambda: HybridTumorClassifier(num_classes=4), DEVICE)

    # =========================================================================
    # EDA MODE
    # =========================================================================
    elif args.mode == "eda":
        data_path = get_data_path()
        train_ds, val_ds, test_ds = get_datasets(data_path)
        train_loader, _, _ = get_dataloaders(train_ds, val_ds, test_ds, batch_size=32)
        print("Generating EDA plots...")
        imshow_samples(train_loader, title="Train_Samples")
        plot_class_distribution(train_ds, val_ds, test_ds, CLASS_NAMES)
        print("EDA plots saved to visualizations/")

    # =========================================================================
    # COMPARE MODE
    # =========================================================================
    elif args.mode == "compare":
        all_results = {}
        pattern = os.path.join(EXPERIMENTS_DIR, "cv_results_*.pkl")
        files = glob.glob(pattern)
        
        if not files:
            print("No CV results found. Run --mode cv --model [Name] first.")
            return

        for f in files:
            res = torch.load(f)
            all_results.update(res)
            
        if len(all_results) < 2:
            print(f"Found {len(all_results)} models ({list(all_results.keys())}). Need at least 2 for comparison.")
            print("Run CV for more models: python main.py --mode cv --model resnet50")
        else:
            compare_models(all_results)

    # =========================================================================
    # CROSS_EVAL MODE (Evaluate on external datasets)
    # =========================================================================
    elif args.mode == "cross_eval":
        print("="*60)
        print("CROSS-DATASET EVALUATION")
        print("="*60)
        
        # Class name mapping for different datasets
        # Our model: ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        # Some datasets use: ['glioma', 'meningioma', 'notumor', 'pituitary'] or similar
        
        # Load trained models
        print("\nLoading trained models...")
        hybrid_path = os.path.join(EXPERIMENTS_DIR, "hybrid_model.pth")
        
        models_to_eval = {}
        if os.path.exists(hybrid_path):
            model = HybridTumorClassifier(num_classes=4).to(DEVICE)
            model.load_state_dict(torch.load(hybrid_path, map_location=DEVICE))
            models_to_eval['Hybrid'] = model
            print("  Loaded Hybrid")
        
        # TinyHybrid
        tiny_path = os.path.join(EXPERIMENTS_DIR, "tinyhybrid_model.pth")
        if os.path.exists(tiny_path):
            model = TinyHybrid(num_classes=4).to(DEVICE)
            model.load_state_dict(torch.load(tiny_path, map_location=DEVICE))
            models_to_eval['TinyHybrid'] = model
            print("  Loaded TinyHybrid")
        
        # Best performing ablations (1 channel)
        best_ablations = [
            ("Hybrid: Optimized", "hybrid:_optimized_model.pth"),
            ("Tiny: No All Attn", "tiny:_no_all_attn_model.pth")
        ]
        for abl_name, abl_file in best_ablations:
            path = os.path.join(EXPERIMENTS_DIR, abl_file)
            if os.path.exists(path):
                try:
                    model = get_ablation_model(abl_name, num_classes=4).to(DEVICE)
                    model.load_state_dict(torch.load(path, map_location=DEVICE))
                    models_to_eval[abl_name] = model
                    print(f"  Loaded {abl_name}")
                except Exception as e:
                    print(f"  Failed to load {abl_name}: {e}")
        
        # Also load ALL trained baselines
        baselines = get_baselines(num_classes=4)
        for baseline_name in baselines.keys():
            safe_name = baseline_name.lower().replace(" ", "_")
            path = os.path.join(EXPERIMENTS_DIR, f"{safe_name}_model.pth")
            if os.path.exists(path):
                try:
                    model = baselines[baseline_name].to(DEVICE)
                    model.load_state_dict(torch.load(path, map_location=DEVICE))
                    models_to_eval[baseline_name] = model
                    print(f"  Loaded {baseline_name}")
                except:
                    pass
        
        if not models_to_eval:
            print("No trained models found! Run --mode pipeline first.")
            return
        
        # External datasets to evaluate
        external_datasets = [
            ("Sartaj", get_sartaj_data_path, 1),  # (name, download_fn, channels for hybrid)
        ]
        
        all_results = []
        
        for dataset_name, download_fn, _ in external_datasets:
            print(f"\n--- Evaluating on {dataset_name} Dataset ---")
            data_path = download_fn()
            if data_path is None:
                print(f"  Failed to download {dataset_name}, skipping...")
                continue
            
            # Load with 1 channel for Hybrid models
            try:
                test_ds_1ch = get_external_test_dataset(data_path, channels=1)
                _, _, test_loader_1ch = get_dataloaders(test_ds_1ch, test_ds_1ch, test_ds_1ch, batch_size=args.batch_size)
            except Exception as e:
                print(f"  Error loading 1-channel data: {e}")
                continue
            
            # Load with 3 channels for baselines
            try:
                test_ds_3ch = get_external_test_dataset(data_path, channels=3)
                _, _, test_loader_3ch = get_dataloaders(test_ds_3ch, test_ds_3ch, test_ds_3ch, batch_size=args.batch_size)
            except Exception as e:
                test_loader_3ch = None
            
            # Evaluate each model
            for model_name, model in models_to_eval.items():
                # Hybrid-based models use 1 channel, baselines use 3 channels
                is_hybrid = model_name in ['Hybrid', 'TinyHybrid'] or model_name.startswith('Hybrid:') or model_name.startswith('Tiny:')
                loader = test_loader_1ch if is_hybrid else test_loader_3ch
                
                if loader is None:
                    continue
                
                try:
                    acc, report, auc, cm, per_class = test_and_report(model, loader, DEVICE, CLASS_NAMES)
                    print(f"  {model_name}: Acc={acc:.4f}, AUC={auc:.4f}")
                    all_results.append({
                        'Dataset': dataset_name,
                        'Model': model_name,
                        'Accuracy': acc,
                        'AUC': auc
                    })
                except Exception as e:
                    print(f"  {model_name} failed: {e}")
        
        # Save and display results
        if all_results:
            df = pd.DataFrame(all_results)
            df.to_csv(os.path.join(EXPERIMENTS_DIR, "cross_eval_results.csv"), index=False)
            print("\n" + "="*60)
            print("CROSS-DATASET RESULTS SUMMARY")
            print("="*60)
            print(df.to_string(index=False))
            print(f"\nResults saved to {EXPERIMENTS_DIR}/cross_eval_results.csv")

if __name__ == "__main__":
    main()
