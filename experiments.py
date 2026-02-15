import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
import glob

import custom_SLIC
import open_cv_SLIC
import metrics
import dataloader

def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.01 
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss * 255
    return np.clip(noisy, 0, 255).astype(np.uint8)

def change_contrast(image):
    alpha = 1.5 
    beta = 0    
    adjusted = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def get_boundary(labels):
    kernel = np.ones((3,3), np.uint8)
    dilated = cv.dilate(labels.astype(np.uint8), kernel, iterations=1)
    boundary = (dilated != labels).astype(np.uint8) * 255
    return boundary

def run_experiment_on_image(image_path, K_vals, m_vals, gt_segments=None):
    results = {'br': [], 'ue': [], 'asa': [], 'time': [], 'k': [], 'method': [], 'variation': []}
    
    original = cv.imread(image_path)
    if original is None:
        print(f"Failed to load {image_path}")
        return results

    variations = [('Original', original), 
                  ('Noise', add_noise(original)), 
                  ('Contrast', change_contrast(original))]
    
    temp_path = "temp_proc.png"
    
    for var_name, img in variations:
        print(f"  Running variation: {var_name}")
        cv.imwrite(temp_path, img)
        
        # 1. Custom SLIC
        for K in K_vals:
            for m in m_vals:
                start = time.time()
                labels, mask = custom_SLIC.SLIC(temp_path, K, m)
                dur = time.time() - start
                
                results['method'].append(f'Custom_m{m}')
                results['k'].append(K)
                results['time'].append(dur)
                results['variation'].append(var_name)
                
                if gt_segments:
                    br_list, ue_list, asa_list = [], [], []
                    for seg in gt_segments:
                        gt_bound = get_boundary(seg)
                        br_list.append(metrics.boundary_recall(gt_bound, mask))
                        ue_list.append(metrics.undersegmentation_error(seg, labels))
                        asa_list.append(metrics.achievable_segmentation_accuracy(seg, labels))
                    results['br'].append(np.mean(br_list))
                    results['ue'].append(np.mean(ue_list))
                    results['asa'].append(np.mean(asa_list))
                else:
                    results['br'].append(0); results['ue'].append(0); results['asa'].append(0)

        # 2. OpenCV SLIC
        for K in K_vals:
            start = time.time()
            labels, mask = open_cv_SLIC.run_opencv_slic(temp_path, K, 10)
            if labels is not None:
                dur = time.time() - start
                results['method'].append('OpenCV_SLIC')
                results['k'].append(K)
                results['time'].append(dur)
                results['variation'].append(var_name)
                
                if gt_segments:
                    br_list, ue_list, asa_list = [], [], []
                    for seg in gt_segments:
                        gt_bound = get_boundary(seg)
                        br_list.append(metrics.boundary_recall(gt_bound, mask))
                        ue_list.append(metrics.undersegmentation_error(seg, labels))
                        asa_list.append(metrics.achievable_segmentation_accuracy(seg, labels))
                    results['br'].append(np.mean(br_list))
                    results['ue'].append(np.mean(ue_list))
                    results['asa'].append(np.mean(asa_list))
                else:
                    results['br'].append(0); results['ue'].append(0); results['asa'].append(0)

        # 3. OpenCV SEEDS
        for K in K_vals:
            start = time.time()
            labels, mask = open_cv_SLIC.run_opencv_seeds(temp_path, K, 4, 2)
            if labels is not None:
                dur = time.time() - start
                results['method'].append('SEEDS')
                results['k'].append(K)
                results['time'].append(dur)
                results['variation'].append(var_name)
                
                if gt_segments:
                    br_list, ue_list, asa_list = [], [], []
                    for seg in gt_segments:
                        gt_bound = get_boundary(seg)
                        br_list.append(metrics.boundary_recall(gt_bound, mask))
                        ue_list.append(metrics.undersegmentation_error(seg, labels))
                        asa_list.append(metrics.achievable_segmentation_accuracy(seg, labels))
                    results['br'].append(np.mean(br_list))
                    results['ue'].append(np.mean(ue_list))
                    results['asa'].append(np.mean(asa_list))
                else:
                    results['br'].append(0); results['ue'].append(0); results['asa'].append(0)

        # 4. Graph Segmentation (Fixed parameters, doesn't depend on K directly, but good for comparison)
        # We can run it once per variation or try to match K. 
        # Graph segmentation depends on sigma, k, min_size. We'll run one config.
        start = time.time()
        labels, mask = open_cv_SLIC.run_graph_segmentation(temp_path, 0.5, 200, 50)
        if labels is not None:
            dur = time.time() - start
            results['method'].append('GraphSeg')
            results['k'].append(0) # Not applicable
            results['time'].append(dur)
            results['variation'].append(var_name)
            
            if gt_segments:
                br_list, ue_list, asa_list = [], [], []
                for seg in gt_segments:
                    gt_bound = get_boundary(seg)
                    br_list.append(metrics.boundary_recall(gt_bound, mask))
                    ue_list.append(metrics.undersegmentation_error(seg, labels))
                    asa_list.append(metrics.achievable_segmentation_accuracy(seg, labels))
                results['br'].append(np.mean(br_list))
                results['ue'].append(np.mean(ue_list))
                results['asa'].append(np.mean(asa_list))
            else:
                results['br'].append(0); results['ue'].append(0); results['asa'].append(0)

    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    return results

def plot_metric(results, metric_key, ylabel, title_suffix, filename):
    plt.figure(figsize=(10, 6))
    
    data = zip(results['method'], results['k'], results['variation'], results[metric_key])
    # Filter for Original variation for the main plot
    # Or plot separate lines.
    
    variations = sorted(list(set(results['variation'])))
    
    for var in variations:
        # Create a subplot or just plot 'Original' for now to keep it clean, 
        # or separate plots per variation.
        # Let's do separate plots for clarity if multiple exist.
        if var != 'Original': continue 
        
        methods = sorted(list(set(results['method'])))
        for m in methods:
            if m == 'GraphSeg': continue # Point comparison, maybe add as hline
            
            # Select data for this method and variation
            subset_idxs = [i for i, v in enumerate(results['variation']) if v == var and results['method'][i] == m]
            ks = [results['k'][i] for i in subset_idxs]
            vals = [results[metric_key][i] for i in subset_idxs]
            
            # Sort by K
            sorted_pairs = sorted(zip(ks, vals))
            if not sorted_pairs: continue
            ks, vals = zip(*sorted_pairs)
            
            plt.plot(ks, vals, label=f"{m} ({var})", marker='o')

        # Add GraphSeg as a line or point if it exists
        gs_idxs = [i for i, v in enumerate(results['variation']) if v == var and results['method'][i] == 'GraphSeg']
        if gs_idxs:
             val = results[metric_key][gs_idxs[0]]
             plt.axhline(y=val, color='r', linestyle='--', label=f'GraphSeg ({var})')

    plt.xlabel('K (Number of Superpixels)')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} vs K {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    K_values = [50, 100, 200] 
    m_values = [10, 20] 
    
    bsds_path = "BSDS500/data" 
    if os.path.exists(bsds_path):
        data = dataloader.load_bsds500(bsds_path, 'test')[:5] 
    else:
        print("BSDS500 not found, using natural/local images.")
        data = dataloader.load_natural_images('.')
        # Filter strictly
        data = [d for d in data if d['filename'].lower().endswith(('.tiff', '.jpg', '.jpeg', '.png'))]
        # Avoid processing result plots
        data = [d for d in data if not d['filename'].startswith('plot_') and not d['filename'].startswith('runtime_')]
        
        # Prefer lena if available for consistency
        lena = [d for d in data if 'lena' in d['filename']]
        if lena:
            data = lena
        else:
            data = data[:1] if data else []

    if not data:
        print("No images found.")
        exit()

    for item in data:
        print(f"Processing {item['filename']}...")
        gt = item.get('ground_truth')
        res = run_experiment_on_image(item['image_path'], K_values, m_values, gt)
        
        # Plot Metrics
        if len(res['k']) > 0:
            plot_metric(res, 'time', 'Runtime (s)', f"({item['filename']})", f"plot_runtime_{item['filename']}.png")
            if gt:
                plot_metric(res, 'br', 'Boundary Recall', f"({item['filename']})", f"plot_br_{item['filename']}.png")
                plot_metric(res, 'ue', 'Undersegmentation Error', f"({item['filename']})", f"plot_ue_{item['filename']}.png")
                plot_metric(res, 'asa', 'ASA', f"({item['filename']})", f"plot_asa_{item['filename']}.png")
            
            print(f"Generated plots for {item['filename']}")
