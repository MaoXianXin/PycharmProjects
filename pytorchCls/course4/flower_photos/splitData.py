import splitfolders  # or import split_folders
import os

if not os.path.exists("/home/mao/Downloads/datasets/flowerDatasets"):
    os.makedirs("/home/mao/Downloads/datasets/flowerDatasets", exist_ok=True)

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio("/home/mao/Downloads/datasets/flower_photos", output="/home/mao/Downloads/datasets/flowerDatasets", seed=1337, ratio=(0.8, 0.1, 0.1), group_prefix=None) # default values
