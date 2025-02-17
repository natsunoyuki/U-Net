# UNet
Implementation of U-Net from https://arxiv.org/pdf/1505.04597.

# Training Data Preparation
Training images and masks should be prepared in the form of `jpg` and `png` images respectively. Segmentation masks should not be saved as `jpg` files. Split the training and test images into separate subdirectories.
```
datasets/
├── data_dir_1
├── ...
└── data_dir_N
    ├── train_images/
    │   └── *.jpg
    ├── train_masks/
    │   └── *.png
    ├── test_images/
    │   └── *.jpg
    └── test_masks/
        └── *.png
```

# Model Training
For the time being, please refer to `notebooks/train_unet.ipynb` for a demonstration of how to train the UNet model. The training pipeline is a work in progress.
