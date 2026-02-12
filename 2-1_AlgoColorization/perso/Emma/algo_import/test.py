import os

# imgs = sorted(os.listdir("dataset/images"))
# masks = sorted(os.listdir("dataset/bw"))

# assert imgs == masks, "Images et masques ne correspondent pas"
# print("Dataset cohérent")

train_imgs = sorted(os.listdir("dataset/train/images"))
train_masks = sorted(os.listdir("dataset/train/bw"))
val_imgs = sorted(os.listdir("dataset/val/images"))
val_masks = sorted(os.listdir("dataset/val/bw"))

assert train_imgs == train_masks
assert val_imgs == val_masks
assert set(train_imgs).isdisjoint(set(val_imgs))
