import os
import shutil
import random

def main():
    # SOURCE (Tout est ici actuellement)
    TRAIN_ROOT = "data/train"
    TRAIN_COLOR = os.path.join(TRAIN_ROOT, "color")
    TRAIN_GRAY = os.path.join(TRAIN_ROOT, "gray")

    # DESTINATION (Les 20% vont ici)
    TEST_ROOT = "data/test"
    TEST_COLOR = os.path.join(TEST_ROOT, "color")
    TEST_GRAY = os.path.join(TEST_ROOT, "gray")

    if not os.path.exists(TRAIN_COLOR):
        print("Dossier train vide. Lance step1 d'abord.")
        return

    # Création des dossiers de test
    os.makedirs(TEST_COLOR, exist_ok=True)
    os.makedirs(TEST_GRAY, exist_ok=True)

    # On liste toutes les images disponibles dans Train/Color
    all_files = sorted([f for f in os.listdir(TRAIN_COLOR) if f.endswith(".tif")])
    total_files = len(all_files)
    
    if total_files == 0:
        print("Aucune image à diviser.")
        return

    # Calcul du nombre d'images à bouger (20%)
    split_count = int(total_files * 0.2)
    
    # Sélection aléatoire
    random.seed(42) # Pour avoir toujours le même split si on relance
    files_to_move = random.sample(all_files, split_count)

    print(f"Déplacement de {split_count} images ({len(files_to_move)/total_files:.0%}) vers le dataset Test...")

    for f in files_to_move:
        # Chemins sources
        src_c = os.path.join(TRAIN_COLOR, f)
        src_g = os.path.join(TRAIN_GRAY, f)

        # Chemins destinations
        dst_c = os.path.join(TEST_COLOR, f)
        dst_g = os.path.join(TEST_GRAY, f)

        # On déplace (move) pour les enlever du train
        if os.path.exists(src_c) and os.path.exists(src_g):
            shutil.move(src_c, dst_c)
            shutil.move(src_g, dst_g)
        else:
            print(f"Attention : Paire incomplète pour {f}, ignorée.")

    print(f"Split terminé.")
    print(f"Train : {len(os.listdir(TRAIN_COLOR))} images")
    print(f"Test  : {len(os.listdir(TEST_COLOR))} images")

if __name__ == "__main__":
    main()