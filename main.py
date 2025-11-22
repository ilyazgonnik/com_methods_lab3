import numpy as np
from PIL import Image
import os

def compress_image(input_path, ratios=[2, 4, 8]):
    img = Image.open(input_path).convert('L')
    matrix = np.array(img)
    h, w = matrix.shape
    original_size = os.path.getsize(input_path)
    U, s, Vt = np.linalg.svd(matrix.astype(np.float32), full_matrices=False)
    
    for ratio in ratios:
        k = max(1, int(original_size / ratio / 2 / (h + w + 1)))
        Uk = U[:,:k].astype(np.float16)
        sk = s[:k].astype(np.float16)
        Vtk = Vt[:k,:].astype(np.float16)
        np.savez_compressed(f'compressed_{ratio}x_{image_path}.npz', U=Uk, s=sk, Vt=Vtk, shape=matrix.shape)

def decompress_image(compressed_path, output_path):
    data = np.load(compressed_path)
    Uk = data['U'].astype(np.float32)
    sk = data['s'].astype(np.float32)
    Vtk = data['Vt'].astype(np.float32)
    rec = (Uk @ np.diag(sk) @ Vtk).clip(0, 255).astype(np.uint8)
    Image.fromarray(rec, 'L').save(output_path)

if __name__ == "__main__":
    images = ['leo', 'ninja', 'steve']
    for image_path in images:
        compress_image(f'{image_path}.bmp')
        decompress_image(f'compressed_2x_{image_path}.npz', f'restored_2x_{image_path}.bmp')
        decompress_image(f'compressed_4x_{image_path}.npz', f'restored_4x_{image_path}.bmp')
        decompress_image(f'compressed_8x_{image_path}.npz', f'restored_8x_{image_path}.bmp')