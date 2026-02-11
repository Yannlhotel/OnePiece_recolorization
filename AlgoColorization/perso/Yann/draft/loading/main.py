import shutil
import os
import step0_scrap, step1_load, step2_split, step3_chunk, step4_lab_convert

def reset_data():
    if os.path.exists("data"):
        # On garde data/loading si nécessaire, sinon on supprime tout
        for folder in ["all_rgb", "train_rgb", "test_rgb", "train", "test", "temp_scraping"]:
            path = os.path.join("data", folder)
            if os.path.exists(path):
                shutil.rmtree(path)

def main():
    print("--- Démarrage Pipeline ---")
    reset_data()
    
    step0_scrap.main()       # Scraping WebP
    step1_load.main()        # WebP -> PNG (RGB)
    step2_split.main()       # Split des PNG
    step3_chunk.main()       # Découpe des PNG
    step4_lab_convert.main() # PNG -> LAB (.npy float32)
    
    print("--- Pipeline Terminée avec Succès ---")

if __name__ == "__main__":
    main()