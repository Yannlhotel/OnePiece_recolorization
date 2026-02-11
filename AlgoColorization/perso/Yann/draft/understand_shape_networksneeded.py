import tifffile as tiff
import numpy as np

def read_tiff_image(file_path):
    try :
        img = tiff.imread(file_path)
        # print(img.shape)
        return img
    except Exception as e:
        print(f"Error reading TIFF image: {e}")
        return None
    
# Example usage
color_path = 'ColorizationOnCluster/data/train/images/chapitre_1_page_002_chunk000.tif'  # Replace with your TIFF image path
gray_path = 'ColorizationOnCluster/data/train/bw/chapitre_1_page_002_chunk000.tif'
color_img = read_tiff_image(color_path)
gray_img = read_tiff_image(gray_path)


# print("Color shape:", color_img.shape, color_img.dtype)
# print("Gray shape:", gray_img.shape, gray_img.dtype)
# print("Color image first line:", color_img[0])
# print("BW image first line:", gray_img[0])


L = color_img[:, :, 0]
print('\n\n',"L", '\n')
print(L )
print(L.max(), L.min(), L.mean())
print('\n\n',"a", '\n')
a = color_img[:, :, 1]
print(a)
print(a.max(), a.min(), a.mean())
print('\n\n',"b", '\n')
b = color_img[:, :, 2]
print(b)
print(b.max(), b.min(), b.mean())
