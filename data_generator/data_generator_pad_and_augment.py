# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:23:04 2023

@author: skalima
"""
from pathlib import Path
import numpy as np
from math import ceil
import random
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
seed_value = 42

"""
Original image size "image_org_size"
will be adjusted to the value given by "image_resizes_to"
by croping and/or padding with mean image value
instead of resizing by interpolation.

Mean and Std of the training set are used for standardization of image.
"""


class train_generator(keras.utils.Sequence):
    """Helper to iterate over the data."""

    def __init__(self, batch_size, image_org_size, image_resizes_to, fnames,
                 augment, Mean, Std,
                 input_dir="images", target_dir="masks"):

        self.batch_size = batch_size
        self.org_size = image_org_size
        self.img_size = image_resizes_to
        self.fnames = fnames
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.Mean = Mean
        self.Std = Std
        self.augmentation = augment

    def __len__(self):
        return ceil(len(self.fnames) / self.batch_size)

    def __getitem__(self, idx):
        """Returns tuple (input, target) of batch size"""
        i = idx * self.batch_size
        hight = min(i + self.batch_size, len(self.fnames))
        batch_img_names = self.fnames[i:hight]

        x = np.zeros((len(batch_img_names),
                     self.img_size[0], self.img_size[1], 1),
                     dtype="float32")
        y = np.zeros((len(batch_img_names),
                     self.img_size[0], self.img_size[1], 1),
                     dtype="uint8")

        # augment the images and masks
        def augmentation_function(image, mask):
            # Random horizontal flipping
            if random.random() >= 0.5:
                image = np.flip(image, axis=-1)
                mask = np.flip(mask, axis=-1)
            # Random vertical flipping
            if random.random() >= 0.5:
                image = np.flip(image, axis=0)
                mask = np.flip(mask, axis=0)
            # Apply brightness (add/subtract a random number from the image)
            if random.random() >= 0.5:
                brightness_factor = random.uniform(-1, 1)
                image = image + brightness_factor
            return np.array(image), np.array(mask)

        for j, fname in enumerate(batch_img_names):
            img = load_img(Path(self.input_dir) / fname,
                           target_size=None, color_mode="grayscale")
            mask = load_img(Path(self.target_dir) / fname,
                            target_size=None, color_mode="grayscale")
            img = np.array(img, dtype="uint8")
            mask = np.array(mask, dtype="uint8")
            im_mean = img.mean()

            # compare size of original image and target size
            dx = int((self.org_size[0] - self.img_size[0]) / 2)
            dy = int((self.org_size[1] - self.img_size[1]) / 2)

            if dx > 0:
                # crop images up and down
                img = img[dx:-dx, :]
                mask = mask[dx:-dx, :]
            if dx < 0:
                # pad images with mean value or zero for masks
                img = np.pad(img, ((abs(dx), abs(dx)), (0, 0)),
                             constant_values=(im_mean, im_mean))
                mask = np.pad(mask, ((abs(dx), abs(dx)), (0, 0)),
                              constant_values=(0, 0))

            if dy > 0:
                # crop images up and down
                img = img[:, dy:-dy]
                mask = mask[:, dy:-dy]
            if dy < 0:
                # pad images with mean value or zero for masks
                img = np.pad(img, ((0, 0), (abs(dy), abs(dy))),
                             constant_values=(im_mean, im_mean))
                mask = np.pad(mask, ((0, 0), (abs(dy), abs(dy))),
                              constant_values=(0, 0))

            assert img.shape == self.img_size, "images size did not match"

            # Normalization: mapping to [0, 1]
            img = np.array(img) / 255.0
            mask = np.array(mask) / 255.0
            # Standardisation of the data
            img = (img - self.Mean)/self.Std
            # Augmentation Raghava style
            if self.augmentation:
                img, mask = augmentation_function(img, mask)

            # save in array
            x[j, :, :, 0] = img
            y[j, :, :, 0] = mask

        return x, y
