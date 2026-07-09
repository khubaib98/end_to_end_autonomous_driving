"""Load a saved steering model and print its input/output signature."""
from __future__ import annotations

import argparse
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .h5 or .keras model file.")
    parser.add_argument("--model-type", default="pilotnet", choices=["pilotnet", "improved", "mobilenet"])
    parser.add_argument("--data-dir", default="data/raw")
    return parser.parse_args()


def main():
    args = parse_args()
    model = tf.keras.models.load_model(args.model, compile=False)
    print("Model loaded successfully")
    print("Model type:", args.model_type)
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
    print("Parameters:", model.count_params())
    print("Data directory:", args.data_dir)


if __name__ == "__main__":
    main()
