# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:22:18 2023

@author: skalima
"""

from tensorflow.keras import layers
from tensorflow import keras

"""
This function Defines U-net Arhitecture.
Input parametars are:
    1) initial number of filters (suggested value is 64))
    2) dropout rate
    3) input image size is a tuple of size 2 because all images are grayscale
    Warning: The image size has to divisible by 16
    this is due to 4x MaxPooling Layers of pool size 2 (2**4)
Not an input parametar:
    - number of upsampling/downsampling layers is alwaays 4
    - number of classes always 2 because we do a simple object sementation
"""


def unet_arhitecture(FLT, LVL, double_CNV, dropout_rate, img_size):

    assert FLT > 1 and FLT < 200, \
        "Initial number of filters should not be less than 4 or more than 200"
    assert dropout_rate >= 0 and dropout_rate < 0.5, \
        "dropout_rate has to be a positive float number in range 0 - 0.5 "
    assert LVL in [2,3,4], \
        "Number of U-net levels has to be 2, 3 or 4"

    initializer = keras.initializers.HeNormal(seed=42)

    # helper function for double conv layer 
    # using Leaky ReLu and Batch Normalization
    def conv_block(x, n_filters):
        """
        One Convolutional block needed for U-net

        Parameters
        ----------
        x : Input Tensor

        n_filters : Number of filters applied in this conv layer

        Returns
        -------
        x: Tensor after 3x3 2D Convolution (stride 1), Batch normalization,
        and activation function that can be ReLu or Leaky ReLu (alpha)
        """

        x = layers.Conv2D(n_filters, kernel_size=3,
                          strides=1, padding="same",
                          kernel_initializer=initializer,
                          activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0)(x)
        return x

    def conv2D_blocks(x, n_filters):
        """
        One or two (if double_CNV = True)
        convolutional blocks followed by Dropout (if dropout_rate > 0)

        Parameters
        ----------
        x : Tensor
        n_filters : Integer
        that defines number of filters used in that particular block

        Returns
        -------
        x : Tensor
        After either 1 or 2 convoutional blocks (3x3) and Dropout defined by Dropout rate

        """
        x = conv_block(x, n_filters)
        if double_CNV:
            x = conv_block(x, n_filters)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        return x

    def downsample_block(x, n_filters):
        """
        Downsample block is MaxPooling followed by 1x or 2x Convolutional block

        Parameters
        ----------
        x : Tensor (N x M x Number of filters)
        That is the input that has to be down-sampled in U-net.
        n_filters : Integer
        that defines number of filters used in that particular block

        Returns
        -------
        x : Downsampled tensor (N/2 x M/2 x Number of filters)
            MaxPoopling followe d by 1 or 2 Convolutional Blocks

        """
        x = layers.MaxPool2D(pool_size=2)(x)
        x = conv2D_blocks(x, n_filters)
        return x

    def upsample_block(x, conv_features, n_filters):
        # upsample
        x = layers.Conv2DTranspose(n_filters, kernel_size=3, strides=2,
                                   padding="same")(x)
        # concatenate
        x = layers.concatenate([x, conv_features])

        # Conv2D twice
        x = conv2D_blocks(x, n_filters)
        return x

    inputs = layers.Input(shape=(img_size[0], img_size[1], 1))

    # encoder: contracting path - downsample
    # First is 2x Convolutional layer
    X0_0 = conv2D_blocks(inputs, FLT)
    # Then is MaxPooling + 2xConvolutional layer dowsampling layers
    X1_0 = downsample_block(X0_0, FLT*(2**1))
    X2_0 = downsample_block(X1_0, FLT*(2**2))

    if LVL == 4:
        # encoder leftovers
        X3_0 = downsample_block(X2_0, FLT*(2**3))
        X4_0 = downsample_block(X3_0, FLT*(2**4))
        # decoder: expanding path - upsample
        X3_1 = upsample_block(X4_0, X3_0, FLT*(2**3))
        X2_2 = upsample_block(X3_1, X2_0, FLT*(2**2))
        X1_3 = upsample_block(X2_2, X1_0, FLT*(2**1))
        X0_4 = upsample_block(X1_3, X0_0, FLT)
        # outputs
        outputs = layers.Conv2D(filters=1, kernel_size=1,
                                padding="same", activation="sigmoid")(X0_4)

    if LVL == 3:
        # encoder leftovers
        X3_0 = downsample_block(X2_0, FLT*(2**3))
        # decoder: expanding path - upsample
        X2_1 = upsample_block(X3_0, X2_0, FLT*(2**2))
        X1_2 = upsample_block(X2_1, X1_0, FLT*(2**1))
        X0_3 = upsample_block(X1_2, X0_0, FLT)
        # outputs
        outputs = layers.Conv2D(filters=1, kernel_size=1,
                                padding="same", activation="sigmoid")(X0_3)

    if LVL == 2:
        # decoder: expanding path - upsample
        X1_1 = upsample_block(X2_0, X1_0, FLT*(2**1))
        X0_2 = upsample_block(X1_1, X0_0, FLT)
        # outputs
        outputs = layers.Conv2D(filters=1, kernel_size=1,
                                padding="same", activation="sigmoid")(X0_2)

        # unet model
    unet_model = keras.Model(inputs, outputs, name="Unet")

    return unet_model