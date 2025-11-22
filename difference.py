import numpy as np
import scipy
from numpy.linalg import svd as numpy_svd
from PIL import Image

def create_special_image(height=533, width=800):
    np.random.seed(42)
    image = np.zeros((height, width), dtype=np.uint8)
    for i in range(0, height, 20):
        image[i:i+10, :] = 128 
    for j in range(0, width, 25):
        image[:, j:j+12] = 128
    
    return image

def compare_svd_results(matrix):
    matrix_float = matrix.astype(np.float64) / 255.0
    U_np, s_np, Vt_np = numpy_svd(matrix_float, full_matrices=False)
    U_sp, s_sp, Vt_sp = scipy.linalg.svd(matrix_float, lapack_driver = "gesvd",full_matrices=False)
    s_np_sorted = np.sort(s_np)
    s_sp_sorted = np.sort(s_sp)
    c_vector = []
    for a, b in zip(s_np_sorted, s_sp_sorted):
        if a == 0 and b == 0:
            c_j = 1.0
        elif a == 0:
            c_j = 0.0
        elif b == 0:
            c_j = 0.0
        else:
            c_j = max(a/b, b/a)
        c_vector.append(c_j) 
    c_vector = np.array(c_vector)
    l2_norm = np.linalg.norm(c_vector)  
    return l2_norm

def main():
    image = create_special_image(533, 800)    
    l2_norm = compare_svd_results(image)
    print(f"L2 норма: {l2_norm}")

if __name__ == "__main__":
    main()