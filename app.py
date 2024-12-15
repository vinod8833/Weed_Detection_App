import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
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
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(img_rgb)

    crop_count = 0
    weed_count = 0
    total_confidence = 0
    box_count = 0

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            confidence = box.conf[0].item()
            total_confidence += confidence
            box_count += 1

            if label == "crop":
                color = (0, 255, 0)
                crop_count += 1
            else:
                color = (0, 0, 255)
                weed_count += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

    average_accuracy = (total_confidence / box_count) * 100 if box_count > 0 else 0

    cv2.putText(frame, f'Average Accuracy: {average_accuracy:.2f}%', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame, crop_count, weed_count, average_accuracy

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/templates/Developers.html')
def developers():
    return render_template('Developers.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    
    if file and allowed_file(file.filename):
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        output_image, crop_count, weed_count, average_accuracy = detect_weeds(img)
        _, buffer = cv2.imencode('.jpg', output_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        response_data = {
            'image': image_base64,
            'crop_count': crop_count,
            'weed_count': weed_count,
            'average_accuracy': f"{average_accuracy:.2f}%"
        }
        print(f"Response - Weed Count: {weed_count}, Crop Count: {crop_count}, Average Accuracy: {average_accuracy:.2f}%")
        return jsonify(response_data)
    else:
        return "Invalid file type", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
