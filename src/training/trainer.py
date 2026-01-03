import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import copy
from src.training.early_stopping import EarlyStopping
from src.config import (EXPERIMENTS_DIR, STATS_DIR, MIN_DELTA, WEIGHT_DECAY,
                        EPOCHS_CUSTOM, LR_CUSTOM, PATIENCE_CUSTOM)

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    y_score = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            probabilities = torch.softmax(outputs, dim=1)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_score.extend(probabilities.cpu().numpy())
            
    return running_loss / total, correct / total, y_true, y_pred, y_score

def train_model(model, train_loader, val_loader, device, name="model", 
                epochs=None, lr=None, patience=None):
    """
    Train a model with early stopping.
    
    Args:
        epochs: If None, uses EPOCHS_CUSTOM from config
        lr: If None, uses LR_CUSTOM from config
        patience: If None, uses PATIENCE_CUSTOM from config
    """
    # Use config defaults if not specified
    epochs = epochs if epochs is not None else EPOCHS_CUSTOM
    lr = lr if lr is not None else LR_CUSTOM
    patience = patience if patience is not None else PATIENCE_CUSTOM
    
    # Label smoothing for regularization (+0.5-1% accuracy)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    
    # Cosine annealing scheduler for smoother convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    scaler = GradScaler('cuda')
    stopper = EarlyStopping(patience=patience, min_delta=MIN_DELTA, mode='max')
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    stats = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Step the learning rate scheduler
        scheduler.step()
        
        stats['train_loss'].append(train_loss)
        stats['train_acc'].append(train_acc)
        stats['val_loss'].append(val_loss)
        stats['val_acc'].append(val_acc)
        
        print(f"[{name}] Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            
        stopper(val_acc)
        if stopper.early_stop:
            print("Early stopping triggered")
            break
            
    model.load_state_dict(best_model_wts)
    
    # Save artifacts to organized folders
    safe_name = name.lower().replace(" ", "_")
    torch.save(model.state_dict(), os.path.join(EXPERIMENTS_DIR, f"{safe_name}.pth"))
    torch.save(stats, os.path.join(STATS_DIR, f"{safe_name}_stats.pkl"))
    
    return model, stats
