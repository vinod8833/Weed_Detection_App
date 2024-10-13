import cv2
import numpy as np
from flask import Flask, render_template, Response, request, send_file, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from io import BytesIO
import base64
from PIL import Image



app = Flask(__name__)
CORS(app)  # Allow CORS for all domains

# Load YOLO model
model = YOLO('weed_detect.pt')  # Replace with your own model path

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Weed detection function for image upload
def detect_weeds(frame):
    # Convert to RGB for YOLO processing
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLO model for weed detection
    results = model(img_rgb)

    # Draw detections on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]  # Get label name
            confidence = box.conf[0].item()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

@app.route('/process_frame', methods=['POST'])
def process_frame():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    
    if file and allowed_file(file.filename):
        # Read image into memory
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Detect weeds in the uploaded frame
        output_image = detect_weeds(img)

        # Encode the output image to JPEG
        _, buffer = cv2.imencode('.jpg', output_image)
        io_buf = BytesIO(buffer)

        # Send the processed image back to the client
        return send_file(io_buf, mimetype='image/jpeg')

    return "Invalid file type", 400

# Route to handle image uploads for weed detection
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    
    if file and allowed_file(file.filename):
        # Read image into memory
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Detect weeds in the uploaded image
        output_image = detect_weeds(img)

        # Encode the output image to JPEG
        _, buffer = cv2.imencode('.jpg', output_image)
        io_buf = BytesIO(buffer)

        # Send the processed image back to the client
        return send_file(io_buf, mimetype='image/jpeg')

    return "Invalid file type", 400

# Main route to render the HTML template
@app.route('/')
def index():
    return render_template('index3.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
