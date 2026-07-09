import argparse
import base64
from io import BytesIO
from pathlib import Path

import cv2
import eventlet
import numpy as np
import socketio
import tensorflow as tf
from flask import Flask
from PIL import Image


def preprocess_pilotnet(image_rgb: np.ndarray) -> np.ndarray:
    image = image_rgb[60:image_rgb.shape[0] - 25, :, :]
    image = cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image.astype(np.float32) / 127.5 - 1.0


def preprocess_mobilenet(image_rgb: np.ndarray) -> np.ndarray:
    image = image_rgb[60:image_rgb.shape[0] - 25, :, :]
    image = cv2.resize(image, (160, 96), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return np.asarray(image, dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a trained steering model with the Udacity simulator."
    )
    parser.add_argument("--model", required=True, help="Path to the .h5 model.")
    parser.add_argument(
        "--model-type",
        choices=["pilotnet", "improved", "mobilenet"],
        default="pilotnet",
    )
    parser.add_argument("--speed-limit", type=float, default=10.0)
    parser.add_argument("--port", type=int, default=4567)
    return parser.parse_args()


args = parse_args()
model_path = Path(args.model)

if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")

print(f"Loading model: {model_path}")
model = tf.keras.models.load_model(model_path, compile=False)
print(f"Loaded model: {model.name}")
print(f"Input shape: {model.input_shape}")
print(f"Parameters: {model.count_params():,}")

preprocessors = {
    "pilotnet": preprocess_pilotnet,
    "improved": preprocess_pilotnet,
    "mobilenet": preprocess_mobilenet,
}
preprocess_image = preprocessors[args.model_type]

sio = socketio.Server(async_mode="eventlet", cors_allowed_origins="*")
flask_app = Flask(__name__)
application = socketio.WSGIApp(sio, flask_app)


def send_control(sid: str, steering_angle: float, throttle: float) -> None:
    sio.emit(
        "steer",
        data={
            "steering_angle": str(float(steering_angle)),
            "throttle": str(float(throttle)),
        },
        to=sid,
    )


@sio.on("telemetry")
def telemetry(sid, data):
    if not data:
        sio.emit("manual", data={}, to=sid)
        return

    try:
        speed = float(data["speed"])

        image = Image.open(
            BytesIO(base64.b64decode(data["image"]))
        ).convert("RGB")
        image_rgb = np.asarray(image)

        processed = preprocess_image(image_rgb)
        batch = np.expand_dims(processed, axis=0)

        prediction = model.predict(batch, verbose=0)
        steering_angle = float(np.asarray(prediction).reshape(-1)[0])

        throttle = 1.0 - speed / args.speed_limit
        throttle = float(np.clip(throttle, -1.0, 1.0))

        print(
            f"steering={steering_angle:+.4f}  "
            f"throttle={throttle:+.4f}  speed={speed:.2f}"
        )

        send_control(sid, steering_angle, throttle)

    except Exception as error:
        print(f"Telemetry error: {error}")
        send_control(sid, 0.0, 0.0)


@sio.on("connect")
def connect(sid, environ, auth=None):
    print(f"Simulator connected: {sid}")
    send_control(sid, 0.0, 0.0)


@sio.on("disconnect")
def disconnect(sid):
    print(f"Simulator disconnected: {sid}")


if __name__ == "__main__":
    print(
        f"Starting server on port {args.port}. "
        "Open the simulator and choose Autonomous Mode."
    )
    eventlet.wsgi.server(
        eventlet.listen(("", args.port)),
        application,
        log_output=False,
    )
