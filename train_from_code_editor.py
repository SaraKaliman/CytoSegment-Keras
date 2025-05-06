# -*- coding: utf-8 -*-
"""
Created on Tue May  6 12:14:04 2025

@author: skalima
"""

from model_training_function import training_function

# define U-net arhitecture and input/output folder
# other parameters are in training_parameters_file.csv
input_folder = r"C:/Users/skalima/Documents/RTDC_Data/TEST_UNET_CODE/input_dir"
output_folder = r"C:/Users/skalima/Documents/RTDC_Data/TEST_UNET_CODE/output_dir"
double_CNV = True
LVL = 2
FLT = 4

# training function
training_function(input_folder, output_folder,
                  double_CNV, LVL, FLT)
