"""Train a PilotNet-style steering regressor.

This is a compact entry point for repository use. The full exploratory workflow,
including visualization and model comparison, is preserved in the notebook.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from src.models import build_pilotnet_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw", help="Directory containing driving_log.csv and IMG/.")
    parser.add_argument("--output-dir", default="models", help="Directory for saved model checkpoints.")
    parser.add_argument("--epochs", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model = build_pilotnet_model()
    model.summary()
    print("This script provides the model entry point. Build tf.data datasets using the notebook or src preprocessing utilities.")
    print(f"Requested data directory: {args.data_dir}")
    print(f"Model checkpoints will be written to: {args.output_dir}")


if __name__ == "__main__":
    main()
