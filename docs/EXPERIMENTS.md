# Experiment Notes

This project evaluates five model families for image-to-steering regression.

## Original PilotNet

The baseline follows the NVIDIA-style behavioral-cloning idea: a stack of convolutional layers learns road features, and dense layers regress a steering angle.

## Improved weighted CNN

This model tests whether stronger regularization and steering-weighted losses improve the baseline. It reduces parameter count but performs worse under RMSE and region-balanced metrics.

## MobileNetV2

MobileNetV2 tests transfer learning. The frozen ImageNet backbone improves average MAE, but it underpredicts rare sharp turns.

## CLIP encoder + regressor

CLIP tests vision-language-pretrained visual representations. The frozen encoder generates 512-dimensional embeddings, and a dense head predicts steering. This produces the best overall MAE but poor sharp-turn performance.

## Tuned PilotNet

PilotNet was tuned using macro regional validation MAE. This sacrifices some straight-driving accuracy but improves rare sharp-turn prediction.
