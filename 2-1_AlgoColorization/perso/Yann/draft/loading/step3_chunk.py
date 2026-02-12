import os
import cv2

def chunk_folder(folder_path, size=512):
    files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    for f in files:
        img_path = os.path.join(folder_path, f)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        count = 0
        for y in range(0, h - size + 1, size):
            for x in range(0, w - size + 1, size):
                chunk = img[y:y+size, x:x+size]
                out_name = f.replace(".png", f"_chunk{count}.png")
                cv2.imwrite(os.path.join(folder_path, out_name), chunk)
                count += 1
        os.remove(img_path) # Supprime l'original pour ne garder que les chunks

def main():
    chunk_folder("data/train_rgb")
    chunk_folder("data/test_rgb")
    print("Chunking terminé.")

if __name__ == "__main__":
    main()