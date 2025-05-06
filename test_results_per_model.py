# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 08:15:59 2024

@author: skalima
"""

import os
from pathlib import Path
from glob import glob
import h5py
from tensorflow import keras
from tensorflow.image import resize_with_crop_or_pad
from metrics.metrics import IoU, dice, precision, recall, FocalTverskyLoss
from time import perf_counter
import numpy as np
import pandas as pd
import cv2
from skimage.color import gray2rgb
from skimage.io import imsave
import argparse

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_folder")
parser.add_argument("--test_set_folder")
parser.add_argument("--plot_results")
args = parser.parse_args()
try:
    double_CNV = eval(args.plot_results)
except:
    print("plot_results argument must be a boolian")
# -----------------------------------------------------------------------------


def test_set_results(model_folder, test_set_folder, plot_results):
    model_folder = Path(model_folder)
    test_set_folder = Path(test_set_folder)
    # Test if folder with test images exists and has correct subfolders
    input_dir = Path("images")
    target_dir = Path("masks")
    if not os.path.isdir(test_set_folder):
        print("Input folder not found:", test_set_folder)
    else:
        os.chdir(test_set_folder)

    if not os.path.isdir(input_dir):
        print(test_set_folder)
        print("Does not contain:", input_dir)

    if not os.path.isdir(target_dir):
        print(test_set_folder)
        print("Does not contain:", target_dir)

    # Test if model folder exists and has one h5 model inside
    if not os.path.isdir(model_folder):
        print("Input folder not found:", model_folder)
    else:
        os.chdir(model_folder)

    model_name = glob("model.*.h5")
    assert len(model_name) == 1, "more than one model found in folder"

    # load keras model with custum objects
    os.chdir(model_folder)
    if plot_results:
        if os.path.isdir("test_set_images"):
            print("Folder test_set_images already exists.")
            print("To perceed delete the folder and re-run the code.")
        else:
            os.mkdir("test_set_images")

    custum = {'FocalTverskyLoss': FocalTverskyLoss,
              'dice': dice,
              'IoU': IoU,
              'recall': recall,
              'precision': precision}
    unet_model = keras.models.load_model(model_folder / model_name[0],
                                         custom_objects=custum)

    f = h5py.File(model_folder / model_name[0], "r")
    Mean = f.attrs["image_mean"]
    Std = f.attrs["image_std"]
    img_size = f.attrs["image_org_size"]
    image_resizes_to = f.attrs["image_resized"]

    # -------------------------------------------------------------------------

    # read test set images
    os.chdir(test_set_folder / Path("images"))
    test_fnames = glob("*.png")
    os.chdir(test_set_folder)

    def read_and_resize_images(img_size, image_resizes_to, fname):
        img = keras.preprocessing.image.load_img(Path("images") / fname,
                                                 target_size=None,
                                                 color_mode="grayscale")
        img = np.array(img, dtype="uint8")
        im_mean = img.mean()
        # compare size of original image and target size
        dx = int((img_size[0] - image_resizes_to[0]) / 2)
        dy = int((img_size[1] - image_resizes_to[1]) / 2)
        if dx > 0:
            # crop images up and down
            img = img[dx:-dx, :]
        if dx < 0:
            # pad images with mean value or zero for masks
            img = np.pad(img, ((abs(dx), abs(dx)), (0, 0)),
                         constant_values=(im_mean, im_mean))
        if dy > 0:
            # crop images up and down
            img = img[:, dy:-dy]
        if dy < 0:
            # pad images with mean image value
            img = np.pad(img, ((0, 0), (abs(dy), abs(dy))),
                         constant_values=(im_mean, im_mean))
        return img

    # read test images normalize, standardize and resize
    N_test = len(test_fnames)
    x = np.zeros((N_test, image_resizes_to[0], image_resizes_to[1], 1),
                 dtype="float32")
    for j, fname in enumerate(test_fnames):
        img = read_and_resize_images(img_size, image_resizes_to, fname)
        img = np.array(img) / 255.0   # normalization: mapping to [0, 1]
        img = (img - Mean)/Std   # standardisation of the data
        x[j, :, :, 0] = img

    t0 = perf_counter()
    y_pred = unet_model.predict(x)
    t1 = perf_counter()
    print(np.round((t1-t0)/N_test, 4), "sec to predict one image")

    def create_bw_image(mask, original_size):
        mask = (mask >= 0.5)*1
        # resize_with_crop_or_pad uses zero padding
        mask = resize_with_crop_or_pad(
            mask, original_size[0], original_size[1])
        mask = np.array(mask, dtype="uint8")
        return mask

    y_pred = create_bw_image(y_pred, img_size)

    def evaluation_matrix(mask_gt, mask_pred):
        e = 10**(-6)
        # Calculate True Positives, False Positives & False Negatives
        TP = np.sum((mask_gt * mask_pred))
        FN = np.sum(((1-mask_pred) * mask_gt))
        FP = np.sum((mask_pred * (1-mask_gt)))
        Dice = 2*TP / (2*TP + FP + FN + e)
        IoUnion = TP/(TP + FP + FN + e)
        Accuracy = TP/np.sum(mask_gt)
        # how many positives are TP
        Precission = TP/(TP+FP)
        # how many of the posivives are been found
        Recall = TP/(TP+FN)
        return Dice, IoUnion, Accuracy, Precission, Recall

    col_names = ["Dice", "IoU", "Accuracy", "Precission", "Recall"]
    all_scores = pd.DataFrame(index=test_fnames, columns=col_names)

    # Save results as png images
    os.chdir(test_set_folder)
    for j, fname in enumerate(test_fnames):
        # read ground truth masks
        mask_gt = keras.preprocessing.image.load_img(Path("masks") / fname,
                                                     target_size=None,
                                                     color_mode="grayscale")
        mask_gt = np.array(mask_gt, dtype="uint8") / 255.0
        mask_pred = y_pred[j, :, :, 0]
        Dice, IoUnion, Accuracy, Precission, Recall = evaluation_matrix(
            mask_gt, mask_pred)
        all_scores.loc[fname, col_names] = evaluation_matrix(
            mask_gt, mask_pred)

        if plot_results:
            image = keras.preprocessing.image.load_img(Path("images") / fname,
                                                       target_size=None,
                                                       color_mode="grayscale")
            image = gray2rgb(image).astype("uint8")
            # calculate diffrence between original and predicted
            mask_diff = mask_gt - mask_pred
            # find contours on mask predicted by a model
            contours, hierarchy = cv2.findContours(mask_pred,
                                                   cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 0, 0), 1)
            image[mask_diff == 1, 0] = 230
            image[mask_diff == -1, 2] = 230

            Dice = np.round(Dice, 4)
            file = fname.split(".")[0] + "_Dice=" + str(Dice) + ".png"
            imsave(model_folder / "test_set_images" / file, image)

    all_scores.loc["Total Average", :] = all_scores.mean()
    all_scores.loc["Total Median", :] = all_scores.median()
    all_scores.loc["Total Std", :] = all_scores.std()

    file = "test_set_results.csv"
    all_scores.to_csv(model_folder / file, sep=";")


if __name__ == "__main__":
    # defined by user
    model_folder = r"C:/Users/skalima/Documents/RTDC_Data/TEST_UNET_CODE/output_dir/U-net_2_LVL_8_FLT_2xconv/LR0.01_BS8_run1"
    test_set_folder = r"C:/Users/skalima/Documents/RTDC_Data/TEST_UNET_CODE/input_dir/test_set"
    plot_results = True
    test_set_results(model_folder, test_set_folder, plot_results)
