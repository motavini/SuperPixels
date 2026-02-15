import cv2 as cv
import numpy as np
import sys

def run_opencv_slic(image_file, K, m):
    image = cv.imread(image_file)
    h, w = image.shape[:2]
    
    region_size = int(np.sqrt((h * w) / K))
    slic = cv.ximgproc.createSuperpixelSLIC(image, region_size=region_size, ruler=m)
    slic.iterate(10)

    labels = slic.getLabels()
    boundary_mask = slic.getLabelContourMask()
    
    return labels, boundary_mask

def run_opencv_seeds(image_file, K, num_levels, prior):
    image = cv.imread(image_file)
    h, w, c = image.shape
    seeds = cv.ximgproc.createSuperpixelSEEDS(w, h, c, K, num_levels, prior, 5, True)
    
    seeds.iterate(image, 10)
    
    labels = seeds.getLabels()
    boundary_mask = seeds.getLabelContourMask()
    
    return labels, boundary_mask

def run_graph_segmentation(image_file, sigma, k, min_size):
    image = cv.imread(image_file)
    gs = cv.ximgproc.segmentation.createGraphSegmentation(sigma, k, min_size)
    labels = gs.processImage(image)
    
    kernel = np.ones((3,3), np.uint8)
    dilated = cv.dilate(labels.astype(np.uint8), kernel, iterations=1)
    boundary_mask = (dilated != labels).astype(np.uint8) * 255
    
    return labels, boundary_mask

if __name__ == "__main__":
    image_file = "lena_color.tiff"
    K = 100
    m = 10
    
    labels_slic, mask_slic = run_opencv_slic(image_file, K, m)
    if labels_slic is not None:
        mask_inv_slic = cv.bitwise_not(mask_slic)
        img_slic = cv.bitwise_and(cv.imread(image_file), cv.imread(image_file), mask=mask_inv_slic)
        cv.imshow("OpenCV SLIC", img_slic)
    
    labels_seeds, mask_seeds = run_opencv_seeds(image_file, K, 4, 2)
    if labels_seeds is not None:
        mask_inv_seeds = cv.bitwise_not(mask_seeds)
        img_seeds = cv.bitwise_and(cv.imread(image_file), cv.imread(image_file), mask=mask_inv_seeds)
        cv.imshow("OpenCV SEEDS", img_seeds)
    
    labels_graph, mask_graph = run_graph_segmentation(image_file, 0.5, 200, 50)
    if labels_graph is not None:
        mask_inv_graph = cv.bitwise_not(mask_graph)
        img_graph = cv.bitwise_and(cv.imread(image_file), cv.imread(image_file), mask=mask_inv_graph)
        cv.imshow("Graph Segmentation", img_graph)
    
    cv.waitKey(0)
    cv.destroyAllWindows()
