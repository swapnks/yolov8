from flask import Flask, request, jsonify, send_from_directory
from waitress import serve
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import base64
import socket
from io import BytesIO

app = Flask(__name__)

model_path = '/Users/skshah/yolo/runs/detect/train16/weights/best.pt'
def fetch_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.254.254.254', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

@app.route("/")
def homepage():
    return send_from_directory('.', 'index.html')

@app.route("/styles.css")
def styles():
    return send_from_directory('.', 'styles.css')

@app.route("/analyze", methods=["POST"])
def analyze_image():
    uploaded_file = request.files["image_file"]
    has_detection, encoded_image, class_index = process_image(uploaded_file.stream)
    return jsonify({"detection_present": has_detection, "image": encoded_image, "label": class_index})

def process_image(image_stream):
    detection_model = YOLO(model_path)
    pil_image = Image.open(image_stream)
    image_array = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    predictions = detection_model.predict(image_array)
    prediction = predictions[0]

    detected_classes = []
    for prediction in predictions:
        detected_boxes = prediction.boxes
        for box in detected_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(image_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
            detected_classes.append(int(box.cls))
            label_text = f"{int(box.cls)} {box.conf.item():.2f}"
            cv2.putText(image_array, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    processed_pil_image = Image.fromarray(image_array)
    buffer = BytesIO()
    processed_pil_image.save(buffer, format="JPEG")
    base64_image = base64.b64encode(buffer.getvalue()).decode()

    primary_label = detected_classes[0] if detected_classes else -1
    return (len(prediction.boxes) > 0, base64_image, primary_label)

if __name__ == '__main__':
    port_number = 8080
    local_ip = fetch_local_ip()
    server_url = f"http://{local_ip}:{port_number}"
    print(f"Server running at: {server_url}")
    serve(app, host='0.0.0.0', port=port_number)
