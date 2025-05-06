# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:35:15 2023
@author: skalima
INPUT:
    a) Running the code in run it in COMMAND LINE uses parser arguments:
    --input_folder: path to folder where images and masks are
    --output_folder: path to folder where training results will be saved

    --num_levels: number of levels (LVL) in U-Net
    --num_filters: initial number of filters (FLT) in U-Net
    --double_conv: will convolution block have 2 conv layers (True / False)

    b) Running the code in run code editor (Pycharm, Spyder, ..) uses
    train_from_code_editor":

    IMPORTANT: csv file "training_parameters_file.csv" contains all other
    traning parameters

OUTPUT:
    1) Model with the best DSC is highlighted in orange in output file
    U-net_*_LVL_*_FLT_2xconv_results.csv
    2) That model is saved in output directory having the same name of
    the best model.


!!! IMPORTANT:
To standardize image values Mean and Std values of the training set
(upon normalization) are used. These values will be saved in output csv file.
This is important information for prespocessing images for the inference.
"""

import os
from pathlib import Path
from glob import glob
import random
from tensorflow.random import set_seed
import numpy as np
import pandas as pd
from datetime import date
import argparse
from tensorflow import keras
import h5py

from metrics.metrics import recall, precision, IoU, dice, FocalTverskyLoss
from unet.various_unet_arhitectures import unet_arhitecture
from data_generator.data_generator_pad_and_augment import train_generator
from saving_output.save_traning_results import save_traning_results

# defining the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--input_folder")
parser.add_argument("--output_folder")
parser.add_argument("--num_levels")
parser.add_argument("--num_filters")
parser.add_argument("--double_conv")
args = parser.parse_args()

# check that arguments were passes correctly
try:
    input_folder = str(args.input_folder)
except AttributeError:
    print("Error: 'input_folder' argument is missing.")
except (TypeError, ValueError):
    print("Error: 'input_folder' must be a valid string.")

try:
    output_folder = str(args.output_folder)
except AttributeError:
    print("Error: 'output_folder' argument is missing.")
except (TypeError, ValueError):
    print("Error: 'output_folder' must be a valid string.")

try:
    double_CNV = eval(args.double_conv)
except AttributeError:
    print("Error: 'double_conv' argument is missing.")
except (TypeError, ValueError):
    print("Error: 'double_conv' must be a boolian (TRUE/FALSE).")

try:
    LVL = int(args.num_levels)
except AttributeError:
    print("Error: 'num_levels' argument is missing.")
except (TypeError, ValueError):
    print("Error: 'num_levels' must be a valid integer (prefered 2-4).")

try:
    FLT = int(args.num_filters)
except AttributeError:
    print("Error: 'num_filters' argument is missing.")
except (TypeError, ValueError):
    print("Error: 'num_filters' must be a valid integer")
    print("(prefered 2**n where n=2-6. e.g. 8, 16, 32).")


def training_function(input_folder, output_folder,
                      double_CNV, LVL, FLT):

    print("input folder will be:", input_folder)
    print("output folder will be:", output_folder)

    print("Number of U-net levels:", LVL)
    print("Initial number of U-net filters (kernels):", FLT)
    print("double convolutional layer:", double_CNV)

    seed_value = 42
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    set_seed(seed_value)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    # read training parameters from csv file
    df_param = pd.read_csv("training_parameters_file.csv", sep=";",
                           index_col=0)

    BATCH_SIZES = eval(df_param.loc["batch sizes", "value"])
    LEARNING_RATE = eval(df_param.loc["learning rates", "value"])
    Num_repeats = int(df_param.loc["experimental repeats", "value"])

    # Define train/val split
    TRAIN_VAL_SPLIT = float(df_param.loc["train-validation split", "value"])
    # Training parameters
    NUM_EPOCHS = int(df_param.loc["number of epochs", "value"])

    # Takes a float smaller than 0.5
    dropout_rate = float(df_param.loc["dropout rate", "value"])
    # Using image augmentation during training
    image_aug = bool(df_param.loc["image augmentation", "value"])

    # Define target image size
    image_org_size = eval(df_param.loc["image original pixel size", "value"])
    image_resizes_to = eval(df_param.loc["image resized to", "value"])
    print("Image target size has to be divisible by", 2**LVL)
    assert image_resizes_to[0] % 2**LVL == 0, \
        "image height is not divisible by 2 to the power of number of levels"
    assert image_resizes_to[1] % 2**LVL == 0, \
        "image width is not divisible by 2 to the power of number of levels"
    print("image original size", image_org_size)
    print("images will be resized to", image_resizes_to)

    # WARNING: to change GAMMA and ALPHA in FocalTverskyLoss
    # go to metrics/metrics.py and change values inside the function
    LOSS = FocalTverskyLoss

    # Define the model name
    model_name = "U-net"
    # include number of U-net levels in the model name
    model_name = model_name + "_" + str(LVL) + "_LVL"
    # include number of filters in model name
    model_name = model_name + "_" + str(FLT) + "_FLT"
    if double_CNV:
        model_name = model_name + "_2xconv"
    else:
        model_name = model_name + "_1xconv"

    # PARAMETERS DEFINED BY A USER
    input_dir = Path("images")
    target_dir = Path("masks")

    # create data frame with all relevant model information
    df = df_param
    df.loc["number of levels", "value"] = str(LVL)
    df.loc["number of filters", "value"] = str(FLT)
    df.loc["double_CNV layer in CNV block", "value"] = str(double_CNV)
    df.loc["input folder", "value"] = str(input_folder)
    df.loc["output folder", "value"] = str(output_folder)

    output_folder = Path(output_folder) / model_name
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    if not os.path.isdir(input_folder):
        print("Input folder not found:", input_folder)
    else:
        os.chdir(input_folder)

    if not os.path.isdir(input_dir):
        print(input_folder)
        print("Does not contain:", input_dir)

    if not os.path.isdir(target_dir):
        print(input_folder)
        print("Does not contain:", target_dir)

    # Calculate means and std of the training set
    # values are used for standardization
    os.chdir(input_dir)
    fnames = sorted(glob("*.png"))
    N = len(fnames)
    x = np.zeros((N, image_org_size[0], image_org_size[1]),
                 dtype="float32")

    for j, fname in enumerate(fnames):
        img = keras.preprocessing.image.load_img(fname,
                                                 target_size=None,
                                                 color_mode="grayscale")
        img = np.array(img) / 255.0  # Normalization: mapping to [0, 1]
        x[j, :, :] = img

    Mean = round(x.mean(), 3)
    Std = round(x.std(), 3)
    df.loc["Norm Images Mean", "value"] = Mean
    df.loc["Norm Images Std", "value"] = Std
    # add date of training in output file
    df.loc["date of training", "value"] = str(date.today())

    fname = model_name + "_parameters.csv"
    df.to_csv(output_folder / fname, sep=";")

    # -------------------------------------------------------------------------

    # load the training data
    os.chdir(input_folder)

    # image names are randomly shuffled
    np.random.seed(seed_value)
    np.random.shuffle(fnames)

    # check if in csv given image size is correct one
    img = keras.preprocessing.image.load_img(Path(input_dir) / fnames[0],
                                             target_size=None,
                                             color_mode="grayscale")
    img = np.array(img)
    assert img.shape == image_org_size, "images don´t have expected size."

    # generating train and Val Sets of data
    N_train = int(len(fnames) * TRAIN_VAL_SPLIT)
    # save names of files used for cross-val
    df_fnames_cross_val = pd.DataFrame(fnames[N_train:],
                                       columns=["names of cross-val files"])
    df_fnames_cross_val.to_csv(Path(output_folder) / "Cross_vall_files.csv",
                               sep=";")

    # -------------------------------------------------------------------------
    # load the model parameters
    unet_model = unet_arhitecture(FLT, LVL, double_CNV, dropout_rate,
                                  image_resizes_to)
    with open(output_folder / 'unet_arhitecture.txt', 'w') as f:
        unet_model.summary(print_fn=lambda x: f.write(x + '\n'))

    def train_model(n_run, LR, BATCH_SIZE):
        os.chdir(input_folder)
        bs_lr_name = "LR" + str(LR) + "_BS" + str(BATCH_SIZE)
        subfolder_name = bs_lr_name + "_run" + str(n_run)
        output_subfolder = output_folder / subfolder_name
        if not os.path.isdir(output_subfolder):
            os.makedirs(output_subfolder)

        # define the learning rate scheduler
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                      factor=0.5,
                                                      patience=10,
                                                      min_lr=0.000001)
        # define all callbacks:
        # LR scheduler, save best model according to Validation Dice
        # and optionally save Tensor Borad
        my_callbacks = [
            reduce_lr,
            keras.callbacks.ModelCheckpoint(filepath=output_subfolder / 'model.{epoch:03d}-{val_dice:.4f}.h5',
                                            save_best_only=True,
                                            monitor="val_dice",
                                            mode='max'),
            # keras.callbacks.TensorBoard(log_dir=output_subfolder / 'logs')
        ]

        # compile using loss and learning rate and metrics
        unet_model.compile(optimizer=keras.optimizers.Adam(
                           learning_rate=LR),
                           loss=LOSS,
                           metrics=[keras.metrics.BinaryAccuracy(threshold=0.5),
                                    dice, IoU, precision, recall]
                           )

        # fit a model using data generator
        history = unet_model.fit(training_generator,
                                 validation_data=validation_generator,
                                 epochs=NUM_EPOCHS,
                                 callbacks=my_callbacks
                                 )

        # save model results
        save_traning_results(input_folder, output_subfolder, history, LOSS)

        # delete all model files except the last one
        # Warning:t this only works for < 1000 epochs
        os.chdir(output_subfolder)
        files = sorted(glob("model.*"))
        for file in files[:-1]:
            os.remove(file)

        # add in hdf5 keras model file as attributes parameters for
        # preprocessing used during inference
        f = h5py.File(files[-1], "r+")
        f.attrs["image_mean"] = Mean
        f.attrs["image_std"] = Std
        f.attrs["image_org_size"] = image_org_size
        f.attrs["image_resized"] = image_resizes_to
        f.flush()
        f.close()
    # -------------------------------------------------------------------------

    # call traning function N-times for all learning rates and batch sizes
    for BATCH_SIZE in BATCH_SIZES:
        training_generator = train_generator(BATCH_SIZE, image_org_size,
                                             image_resizes_to,
                                             fnames[:N_train],
                                             augment=image_aug,
                                             Mean=Mean, Std=Std)
        validation_generator = train_generator(BATCH_SIZE, image_org_size,
                                               image_resizes_to,
                                               fnames[N_train:],
                                               augment=False,
                                               Mean=Mean, Std=Std)
        for LR in LEARNING_RATE:
            for n_run in range(Num_repeats):
                train_model(n_run, LR, BATCH_SIZE)

    # -------------------------------------------------------------------------

    def summarize_training_results(BATCH_SIZES, LEARNING_RATE, Num_repeats):
        os.chdir(output_folder)
        df_results = pd.DataFrame(columns=[])
        for bs in BATCH_SIZES:
            for lr in LEARNING_RATE:
                for n_run in range(Num_repeats):
                    folder = "LR" + str(lr) + "_BS" + str(bs) + "_run" + str(n_run)
                    os.chdir(folder)
                    df = pd.read_csv("training_results.csv", sep=";")
                    os.chdir("..")
                    index_name = "LR: " + str(lr) + " BS: " + str(bs)
                    df_results.loc["max F1 on cross-val run " + str(n_run), index_name] = \
                        round(df["Max_Val_Dice"][0], 6)
        max_val = df_results.max().max()

        def highlight_max(s):
            is_max = s == max_val
            return ['color: darkorange' if v else '' for v in is_max]
        fname = model_name + "_results.xlsx"
        df_results.style.apply(highlight_max).to_excel(fname)

    summarize_training_results(BATCH_SIZES, LEARNING_RATE, Num_repeats)


if __name__ == "__main__":
    training_function(input_folder, output_folder,
                      double_CNV, LVL, FLT)
