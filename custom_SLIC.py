import cv2 as cv
import numpy as np

def SLIC(image_file, K, m):
    image = cv.imread(image_file)

    # Convert to LAB color space
    lab_image = cv.cvtColor(image, cv.COLOR_BGR2LAB).astype(np.float32)

    h, w = image.shape[:2]
    N = h*w
    S = int(np.sqrt(N / K))

    # Normalize L, a, b channels
    L = lab_image[:,:,0] * 100.0 / 255.0
    a = lab_image[:,:,1] - 128.0
    b = lab_image[:,:,2] - 128.0

    centers = init_cluster_centers(L, a, b, S)
    # Number of recommended iterations is 10
    for _ in range(2):
        labels = assign_to_clusters(L, a, b, centers, S, m)
        centers = update_cluster_centers(L, a, b, labels, centers)

    labels = assign_to_clusters(L, a, b, centers, S, m)
    labels = enforce_connectivity(labels)

    print(labels[:3*S, :3*S])  # Debug


def distance(coord_1, coord_2, L1, a1, b1, L2, a2, b2, S, m):
    d_xy = np.sqrt((coord_1[0] - coord_2[0])**2 + (coord_1[1] - coord_2[1])**2)
    d_lab = np.sqrt((L1 - L2)**2 + (a1 - a2)**2 + (b1 - b2)**2)
    return d_lab + (m * d_xy / S)

def assign_to_clusters(L, a, b, centers, S, m):
    h, w = L.shape
    distances = np.full((h, w), np.inf)
    labels = np.full((h, w), -1, dtype=np.int32)

    for idx, (_, _, _, i, j) in enumerate(centers):
        for di in range(-S, S + 1):
            for dj in range(-S, S + 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    d = distance((i, j), (ni, nj), L[ni, nj], a[ni, nj], b[ni, nj], centers[idx][0], centers[idx][1], centers[idx][2], S, m)
                    if d < distances[ni, nj]:
                        distances[ni, nj] = d
                        labels[ni, nj] = idx
    
    return labels

def update_cluster_centers(L, a, b, labels, centers):
    new_centers = []
    for idx in range(len(centers)):
        mask = (labels == idx)
        if np.any(mask):
            L_mean = np.mean(L[mask])
            a_mean = np.mean(a[mask])
            b_mean = np.mean(b[mask])
            i_mean = int(np.mean(np.where(mask)[0]))
            j_mean = int(np.mean(np.where(mask)[1]))
            new_centers.append((L_mean, a_mean, b_mean, i_mean, j_mean))
        else:
            new_centers.append(centers[idx])
    return new_centers


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

def enforce_connectivity(labels):
    label_list = np.unique(labels)
    for label in label_list:
        mask = (labels == label)
        num_disjoint , disjoint_labels = cv.connectedComponents(mask.astype(np.uint8))
        if num_disjoint > 2:
            disjoint_sizes = {j: np.sum(disjoint_labels == j) for j in range(1, num_disjoint)}
            max_size = max(disjoint_sizes.values())
            for j in range(1, num_disjoint):
                if disjoint_sizes[j] < max_size:
                    labels[disjoint_labels == j] = -1

    stray_pixels = (labels == -1)
    label_sizes = {label: np.sum(labels == label) for label in label_list}
    num_stray, stray_labels = cv.connectedComponents(stray_pixels.astype(np.uint8))
    for i in range(1, num_stray):
        mask = (stray_labels == i)
        max_label = largest_neighbour(mask, labels, label_list, label_sizes)
        labels[mask] = max_label

    return labels

def largest_neighbour(mask, labels, label_list, label_sizes):
    max_size = 0
    max_label = -1
    neighbours = neighbour_labels(mask, labels)
    for label in neighbours:
        if label_sizes[label] > max_size:
            max_size = label_sizes[label]
            max_label = label
    return max_label

def neighbour_labels(mask, labels):
    kernel = np.ones((3,3), np.uint8)
    dilated = cv.dilate(mask.astype(np.uint8), kernel, iterations=1)
    border = dilated.astype(bool) & (~mask)
    neighbor_labels = np.unique(labels[border])
    neighbor_labels = neighbor_labels[neighbor_labels != -1]

    return neighbor_labels


if __name__ == "__main__":
    image_file = "lena_color.tiff"
    K = 100 
    m = 10  
    SLIC(image_file, K, m)