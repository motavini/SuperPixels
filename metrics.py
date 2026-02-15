import numpy as np
import time
import cv2 as cv

def boundary_recall(ground_truth_mask, boundary_mask, d=2):
    if np.sum(ground_truth_mask) == 0:
        return 0.0

    gt = (ground_truth_mask > 0).astype(np.uint8)
    bd = (boundary_mask > 0).astype(np.uint8)
    
    dist_map = cv.distanceTransform(1 - bd, cv.DIST_L2, 3)
    
    gt_pixels = (gt == 1)
    hits = np.sum(dist_map[gt_pixels] <= d)
    
    recall = hits / np.sum(gt_pixels)
    return recall

def undersegmentation_error(ground_truth_labels, superpixel_labels):
    N = ground_truth_labels.size
    sp_unique = np.unique(superpixel_labels)
    
    total_error = 0
    
    for s_label in sp_unique:
        s_mask = (superpixel_labels == s_label)
        s_size = np.sum(s_mask)
        
        gt_in_sp = ground_truth_labels[s_mask]
        unique_gt = np.unique(gt_in_sp)
        
        # For each GT region touched by this superpixel, we add |Sj|
        # The formula boils down to sum(|Sj| * num_touched_GT) - N
        # But standard def is per GT region. The sum over GT regions of intersection.
        # Equivalent: term |Sj| is added once for every G_i it intersects.
        
        k_j = len(unique_gt)
        total_error += s_size * k_j
        
    return (total_error - N) / N

def achievable_segmentation_accuracy(ground_truth_labels, superpixel_labels):
    N = superpixel_labels.size
    sp_unique = np.unique(superpixel_labels)
    
    sum_max_overlaps = 0
    
    for s_label in sp_unique:
        s_mask = (superpixel_labels == s_label)
        
        gt_in_sp = ground_truth_labels[s_mask]
        
        if gt_in_sp.size == 0:
            continue
            
        values, counts = np.unique(gt_in_sp, return_counts=True)
        max_overlap = np.max(counts)
        sum_max_overlaps += max_overlap
        
    return sum_max_overlaps / N

def measure_runtime(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Runtime of {func.__name__}: {end - start:.4f} seconds")
        return result
    return wrapper
