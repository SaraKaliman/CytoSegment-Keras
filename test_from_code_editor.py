# -*- coding: utf-8 -*-
"""
Created on Tue May  6 13:20:31 2025

@author: skalima
"""
from model_testing_function import model_testing_function


# defined data folders
model_folder = r"C:/Users/skalima/Documents/RTDC_Data/TEST_UNET_CODE/output_dir/U-net_2_LVL_4_FLT_2xconv/LR0.01_BS16_run1"
test_set_folder = r"C:/Users/skalima/Documents/RTDC_Data/TEST_UNET_CODE/input_dir/test_set"
# define if images with FP (Blue) and FN (red) will be saved
plot_results = True

# testing function
model_testing_function(model_folder, test_set_folder, plot_results)
