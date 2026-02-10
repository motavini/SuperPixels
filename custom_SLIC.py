import cv2 as cv
import numpy as np

def custom_SLIC(image_file, K, m):
    # Read the image
    image = cv.imread(image_file)

    # Convert to LAB color space
    lab_image = cv.cvtColor(image, cv.COLOR_BGR2LAB)

    # Get image dimensions
    h, w = image.shape[:2]
    N = h*w
    print(f"Image dimensions: {h}x{w}, Total pixels: {N}")

if __name__ == "__main__":
    image_file = "lena_color.tiff"
    K = 100 
    m = 10  
    custom_SLIC(image_file, K, m)