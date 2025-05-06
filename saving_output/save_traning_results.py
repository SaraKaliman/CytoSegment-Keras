# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 13:20:21 2023

@author: skalima
"""

import os
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def save_traning_results(Data_folder, output_folder, history, Loss_function_name):
    """
    Helper function that takes location where model_outputs folder will be created
    and model history.
    Function saves plots of model loss, IoU and Dice score as a function of Epoche.
    Also an excel file is saved with this information.
    This information is also available in logs folder for inspection in TensorBoard
    """

    os.chdir(Data_folder)

    # Plots
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel(Loss_function_name)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig(Path(output_folder) / "ModelLoss.png", dpi=150)

    plt.figure()
    plt.plot(history.history['IoU'])
    plt.plot(history.history['val_IoU'])
    plt.title('Model IoU')
    plt.ylabel('IoU')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig(Path(output_folder) / "ModelIoU.png", dpi=150)

    plt.figure()
    plt.plot(history.history['dice'])
    plt.plot(history.history['val_dice'])
    plt.title('Model Dice')
    plt.ylabel('Dice Score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig(Path(output_folder) / "ModelDice.png", dpi=150)

    df = pd.DataFrame()
    df["Learning_Rate"] = history.history['lr']
    df["Loss"] = history.history['loss']
    df["IoU"] = history.history['IoU']
    df["Dice"] = history.history['dice']
    df["Precision"] = history.history['precision']
    df["Recall"] = history.history['recall']
    df["Validation_loss"] = history.history['val_loss']
    df["Validation_IoU"] = history.history['val_IoU']
    df["Validation_Dice"] = history.history['val_dice']
    df["Validation_Precision"] = history.history['val_precision']
    df["Validation_Recall"] = history.history['val_recall']
    
    df.loc[0, "Max_Val_Dice"] = df["Validation_Dice"].max()
    df.loc[0, "Epoch_Max_Val_Dice"] = df["Validation_Dice"].argmax()+1
    
    df.to_csv(Path(output_folder) / "training_results.csv", sep=";")

    