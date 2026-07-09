# Simulator Deployment Notes

The Udacity simulator is an older Windows application that communicates through Socket.IO. Newer Socket.IO server packages may produce an unsupported protocol error. Use the legacy packages in `requirements-simulator.txt` if necessary.

## WSL workflow

```bash
conda create -n selfdriving python=3.11 -y
conda activate selfdriving
pip install -r requirements-simulator.txt
python src/drive_simulator.py --model models/pilotnet_tuned_low_dropout.h5 --model-type pilotnet --speed-limit 6
```

Then open the Windows simulator and select autonomous mode.

## Model preprocessing

- `pilotnet`: crop, resize to 66 × 200, RGB to YUV, normalize to [-1, 1]
- `improved`: same as PilotNet
- `mobilenet`: crop, resize to 96 × 160, MobileNetV2 preprocessing
