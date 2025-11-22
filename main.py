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
        np.savez_compressed(f'compressed_{ratio}x.npz', U=Uk, s=sk, Vt=Vtk, shape=matrix.shape)

def decompress_image(compressed_path, output_path):
    data = np.load(compressed_path)
    Uk = data['U'].astype(np.float32)
    sk = data['s'].astype(np.float32)
    Vtk = data['Vt'].astype(np.float32)
    rec = (Uk @ np.diag(sk) @ Vtk).clip(0, 255).astype(np.uint8)
    Image.fromarray(rec, 'L').save(output_path)

if __name__ == "__main__":
    compress_image('input.bmp')
    decompress_image('compressed_2x.npz', 'restored_2x.bmp')
    decompress_image('compressed_4x.npz', 'restored_4x.bmp')
    decompress_image('compressed_8x.npz', 'restored_8x.bmp')