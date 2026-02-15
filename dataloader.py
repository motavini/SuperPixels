import os
import glob
import cv2 as cv
import numpy as np
import scipy.io

def load_bsds500(bsds_path, subset='test'):
    img_dir = os.path.join(bsds_path, 'images', subset)
    gt_dir = os.path.join(bsds_path, 'groundTruth', subset)
    
    image_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    data = []
    
    for img_path in image_files:
        basename = os.path.basename(img_path).replace('.jpg', '.mat')
        gt_path = os.path.join(gt_dir, basename)
        
        if not os.path.exists(gt_path):
            continue
            
        mat = scipy.io.loadmat(gt_path)
        
        gts = []
        for i in range(mat['groundTruth'].shape[1]):
            seg = mat['groundTruth'][0,i]['Segmentation'][0,0]
            gts.append(seg)
            
        data.append({'image_path': img_path, 'ground_truth': gts, 'filename': basename})
        
    return data

def load_natural_images(image_dir):
    files = sorted(glob.glob(os.path.join(image_dir, '*')))
    data = []
    for f in files:
        img = cv.imread(f)
        if img is not None:
            data.append({'image_path': f, 'filename': os.path.basename(f)})
    return data
