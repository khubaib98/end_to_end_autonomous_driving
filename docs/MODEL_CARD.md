# Model Card

## Intended use

Offline and simulator-based experimentation with end-to-end steering prediction from synthetic road images.

## Not intended for

Real-world autonomous driving or safety-critical deployment.

## Key risks

- Synthetic training data may not transfer to real roads.
- Frame-level metrics do not guarantee closed-loop stability.
- Rare turn and recovery cases require more data.

## Best use of results

Use CLIP or MobileNetV2 to demonstrate improvement in dataset-weighted average MAE, and use tuned PilotNet to demonstrate balanced turn performance.
