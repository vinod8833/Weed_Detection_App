import cv2
import numpy as np
from flask import Flask, render_template, Response, request, send_file, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import base64

app = Flask(__name__)
CORS(app)

model = YOLO('weed_detect.pt')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_weeds(frame):
    # Convert to RGB for YOLO processing
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(img_rgb)

    crop_count = 0
    weed_count = 0

    # Draw detections on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            confidence = box.conf[0].item()

            if label == "crop":
                color = (0, 255, 0)
                crop_count += 1
            else:
                color = (0, 0, 255)
                weed_count += 1

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
    
    return frame, crop_count, weed_count

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    
    if file and allowed_file(file.filename):
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        output_image, crop_count, weed_count  = detect_weeds(img)
        _, buffer = cv2.imencode('.jpg', output_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        response_data = {
            'image': image_base64,
            'crop_count': crop_count,
            'weed_count': weed_count
        }
        print(f"response weed_count {weed_count} crop_count {crop_count}")
        return jsonify(response_data)
    else:
        return "Invalid file type", 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    