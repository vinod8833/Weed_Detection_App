from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
from flask_cors import CORS
from ultralytics import YOLO
import ssl


app = Flask(__name__)
CORS(app)  # Allow CORS for all domains

# Load YOLO model
model = YOLO('weed_detect.pt')  # Replace with your own model path

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Weed detection function
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

@app.route('/')
def index():
    return render_template('indext.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Get the JSON data from the POST request
        data = request.json

        # Decode the base64 image data to get the video frame
        image_data = base64.b64decode(data['frame'])
        np_image = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        # Detect weeds in the frame
        processed_frame = detect_weeds(frame)

        # Encode the processed frame back to base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Return the processed frame as JSON
        return jsonify({'processed_frame': processed_frame_base64})

    except Exception as e:
        return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
#     context.load_cert_chain(certfile='cert.pem', keyfile='key.pem')

#     app.run(host='0.0.0.0',ssl_context=context, debug=True)
    


if __name__ == '__main__':
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile='cert.pem', keyfile='key.pem')
    app.run(ssl_context=context, host='0.0.0.0', port=5000)

