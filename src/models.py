"""Model definitions for end-to-end steering prediction."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    ELU,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    SpatialDropout2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from .preprocessing import INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, TRANSFER_CHANNELS, TRANSFER_HEIGHT, TRANSFER_WIDTH


def build_pilotnet_model(learning_rate: float = 1e-4, dropout_rate: float = 0.5) -> Model:
    inputs = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS), name="input_image")
    x = Conv2D(24, (5, 5), strides=(2, 2), activation="elu", name="conv_1")(inputs)
    x = Conv2D(36, (5, 5), strides=(2, 2), activation="elu", name="conv_2")(x)
    x = Conv2D(48, (5, 5), strides=(2, 2), activation="elu", name="conv_3")(x)
    x = Conv2D(64, (3, 3), activation="elu", name="conv_4")(x)
    x = Conv2D(64, (3, 3), activation="elu", name="conv_5")(x)
    x = Flatten(name="flatten")(x)
    x = Dense(100, activation="elu", name="dense_100")(x)
    x = Dropout(dropout_rate, name="dropout_1")(x)
    x = Dense(50, activation="elu", name="dense_50")(x)
    x = Dropout(dropout_rate * 0.5, name="dropout_2")(x)
    x = Dense(10, activation="elu", name="dense_10")(x)
    outputs = Dense(1, activation="linear", name="steering_output")(x)
    model = Model(inputs, outputs, name="PilotNet")
    model.compile(
        optimizer=Adam(learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"), tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def build_tunable_pilotnet(
    learning_rate: float = 3e-4,
    dropout_rate: float = 0.25,
    l2_strength: float = 1e-5,
    dense_width: int = 100,
) -> Model:
    inputs = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS), name="input_image")
    x = Conv2D(24, 5, strides=2, activation="elu", kernel_regularizer=l2(l2_strength))(inputs)
    x = Conv2D(36, 5, strides=2, activation="elu", kernel_regularizer=l2(l2_strength))(x)
    x = Conv2D(48, 5, strides=2, activation="elu", kernel_regularizer=l2(l2_strength))(x)
    x = Conv2D(64, 3, activation="elu", kernel_regularizer=l2(l2_strength))(x)
    x = Conv2D(64, 3, activation="elu", kernel_regularizer=l2(l2_strength))(x)
    x = Flatten()(x)
    x = Dense(dense_width, activation="elu", kernel_regularizer=l2(l2_strength))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(50, activation="elu", kernel_regularizer=l2(l2_strength))(x)
    x = Dropout(dropout_rate * 0.5)(x)
    x = Dense(10, activation="elu", kernel_regularizer=l2(l2_strength))(x)
    outputs = Dense(1, activation="linear")(x)
    model = Model(inputs, outputs, name="TunedPilotNet")
    model.compile(optimizer=Adam(learning_rate), loss="mse", metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")])
    return model


def _conv_block(inputs, filters, kernel_size, strides, l2_strength, spatial_dropout=0.0, name="block"):
    x = Conv2D(filters, kernel_size, strides=strides, padding="valid", use_bias=False, kernel_regularizer=l2(l2_strength), name=f"{name}_conv")(inputs)
    x = BatchNormalization(name=f"{name}_bn")(x)
    x = ELU(name=f"{name}_elu")(x)
    if spatial_dropout > 0:
        x = SpatialDropout2D(spatial_dropout, name=f"{name}_spatial_dropout")(x)
    return x


def build_improved_cnn(learning_rate: float = 3e-4, l2_strength: float = 1e-5, dropout_rate: float = 0.25) -> Model:
    inputs = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS), name="input_image")
    x = _conv_block(inputs, 24, 5, 2, l2_strength, name="block_1")
    x = _conv_block(x, 36, 5, 2, l2_strength, name="block_2")
    x = _conv_block(x, 48, 5, 2, l2_strength, spatial_dropout=0.05, name="block_3")
    x = _conv_block(x, 64, 3, 1, l2_strength, spatial_dropout=0.05, name="block_4")
    x = _conv_block(x, 64, 3, 1, l2_strength, name="block_5")
    x = GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = Dense(64, activation="elu", kernel_regularizer=l2(l2_strength), name="dense_64")(x)
    x = Dropout(dropout_rate, name="dropout")(x)
    x = Dense(16, activation="elu", kernel_regularizer=l2(l2_strength), name="dense_16")(x)
    outputs = Dense(1, activation="linear", name="steering_output")(x)
    model = Model(inputs, outputs, name="ImprovedWeightedCNN")
    model.compile(optimizer=Adam(learning_rate), loss=tf.keras.losses.Huber(delta=0.15), metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")])
    return model


def build_mobilenet_regressor(learning_rate: float = 3e-4, dropout_rate: float = 0.30, l2_strength: float = 1e-5) -> Model:
    inputs = Input(shape=(TRANSFER_HEIGHT, TRANSFER_WIDTH, TRANSFER_CHANNELS), name="road_image")
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(TRANSFER_HEIGHT, TRANSFER_WIDTH, TRANSFER_CHANNELS), include_top=False, weights="imagenet"
    )
    backbone.trainable = False
    x = backbone(inputs, training=False)
    x = GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = Dense(128, activation="relu", kernel_regularizer=l2(l2_strength), name="regression_dense_128")(x)
    x = Dropout(dropout_rate, name="regression_dropout")(x)
    x = Dense(32, activation="relu", kernel_regularizer=l2(l2_strength), name="regression_dense_32")(x)
    outputs = Dense(1, activation="linear", name="steering_output")(x)
    model = Model(inputs, outputs, name="MobileNetV2SteeringRegressor")
    model.compile(optimizer=Adam(learning_rate), loss=tf.keras.losses.Huber(delta=0.15), metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")])
    return model
