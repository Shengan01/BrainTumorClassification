import torch
import time
import psutil
import os
import numpy as np
from statistics import mean, stdev

def measure_model_metrics(model_class, device):
    print("Measuring Hybrid Model Performance Metrics...")
    
    memory_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    
    # Measure Load Time (requires re-init to capture load)
    torch.cuda.synchronize()
    start_time = time.time()
    
    model = model_class().to(device)
    
    torch.cuda.synchronize()
    model_load_time = (time.time() - start_time) * 1000
    
    memory_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    memory_usage = memory_after - memory_before
    
    model.eval()
    times_single = []
    times_batch = []
    prob_distributions = []
    
    single_input = torch.randn(1, 1, 224, 224).to(device)
    batch_input = torch.randn(32, 1, 224, 224).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(single_input)
    torch.cuda.synchronize()

    # Single Inference
    for _ in range(100):
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            output = model(single_input)
            probs = torch.nn.functional.softmax(output, dim=1)
        torch.cuda.synchronize()
        times_single.append((time.time() - start_time) * 1000)
        prob_distributions.append(probs[0].cpu().numpy())
    
    # Batch Inference
    for _ in range(10):
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            _ = model(batch_input)
        torch.cuda.synchronize()
        times_batch.append((time.time() - start_time) * 1000)
    
    avg_single = mean(times_single)
    std_single = stdev(times_single)
    avg_batch = mean(times_batch)
    std_batch = stdev(times_batch)
    
    print(f"\nModel Loading Time: {model_load_time:.2f} ms")
    print(f"Memory Usage: {memory_usage:.2f} MB")
    print(f"\nSingle Image Inference:")
    print(f"  Average: {avg_single:.2f} ms ± {std_single:.2f} ms")
    print(f"  Throughput: {1000/avg_single:.2f} images/second")
    print(f"\nBatch Inference (32 images):")
    print(f"  Average: {avg_batch:.2f} ms ± {std_batch:.2f} ms")
    print(f"  Throughput: {32000/avg_batch:.2f} images/second")
    
    avg_probs = np.mean(prob_distributions, axis=0)
    print("\nAverage Probability Distribution (random input):")
    # Assuming standard class order
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'] 
    for class_name, prob in zip(class_names, avg_probs):
        print(f"  {class_name}: {prob*100:.2f}%")
    
    return {
        'model_load_time': model_load_time,
        'memory_usage': memory_usage,
        'single_inference': avg_single,
        'single_inference_std': std_single,
        'batch_inference': avg_batch,
        'batch_inference_std': std_batch,
        'prob_distribution': avg_probs
    }
