import cv2 as cv
import numpy as np

def custom_SLIC(image_file, K, m):
    # Read the image
    image = cv.imread(image_file)

    # Convert to LAB color space
    lab_image = cv.cvtColor(image, cv.COLOR_BGR2LAB).astype(np.float32)

    h, w = image.shape[:2]
    N = h*w
    S = int(np.sqrt(N / K))

    L = lab_image[:,:,0] * 100.0 / 255.0
    a = lab_image[:,:,1] - 128.0
    b = lab_image[:,:,2] - 128.0

    centers = init_cluster_centers(L, a, b, S)


def distance(L1, a1, b1, L2, a2, b2, S, m):
    dc = np.sqrt((L1 - L2)**2 + (a1 - a2)**2 + (b1 - b2)**2)
    ds = np.sqrt((L1 - L2)**2 + (a1 - a2)**2 + (b1 - b2)**2) / S
    return np.sqrt(dc**2 + (m * ds)**2)

def gradient_magnitude(L, a, b):
    grad_L_x, grad_L_y = np.gradient(L)
    grad_a_x, grad_a_y = np.gradient(a)
    grad_b_x, grad_b_y = np.gradient(b)

    grad_magnitude = np.sqrt(grad_L_x**2 + grad_L_y**2 + grad_a_x**2 + grad_a_y**2 + grad_b_x**2 + grad_b_y**2)
    return grad_magnitude

def init_cluster_centers(L, a, b, S):
    h, w = L.shape
    center = []
    for i in range(S//2, h, S):
        for j in range(S//2, w, S):
            center.append((L[i,j], a[i,j], b[i,j], i, j))

    grad_norm = gradient_magnitude(L, a, b)
    for idx, (_, _, _, i, j) in enumerate(center):
        min_grad = grad_norm[i, j]
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    if grad_norm[ni, nj] < min_grad:
                        center[idx] = (L[ni, nj], a[ni, nj], b[ni, nj], ni, nj)
                        min_grad = grad_norm[ni, nj]

    return center


if __name__ == "__main__":
    image_file = "lena_color.tiff"
    K = 100 
    m = 10  
    custom_SLIC(image_file, K, m)