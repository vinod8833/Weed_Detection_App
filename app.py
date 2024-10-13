import os
import cv2
import torch
import numpy as np
from flask import Flask, redirect, url_for, request, render_template, Response, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

# YOLO model path (update this to your model's path)
model = YOLO('weed_detect.pt')

# Define upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Reusable weed detection function for both video frames and uploaded images
def detect_weeds(input_data):
    if isinstance(input_data, str):
        # If the input is a file path (image upload)
        img = cv2.imread(input_data)
    else:
        # If the input is a frame from a video
        img = input_data
    
    # Convert to RGB as required by the YOLO model
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run inference using YOLO model
    results = model(img_rgb)

    # Plot detections on the image/frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]  # Get label name
            confidence = box.conf[0].item()
            # Draw bounding box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'{label} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return img

# Route for video feed (uses webcam frames)
def generate_frames(camera_type='back'):
    # Select the camera based on the camera type
    camera_index = 0 if camera_type == 'back' else 1  # Assume 0 is back camera and 1 is front camera
    cap = cv2.VideoCapture(camera_index)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Detect weeds in the frame
            frame = detect_weeds(frame)
            
            # Encode the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame to be displayed on the frontend
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed/<camera_type>')
def video_feed(camera_type):
    return Response(generate_frames(camera_type), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to handle the image upload
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run weed detection on the uploaded image
        output_image = detect_weeds(filepath)
        
        # Save the output image with detections
        output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_' + filename)
        cv2.imwrite(output_image_path, output_image)
        
        # Return the processed image
        return redirect(url_for('uploaded_file', filename=os.path.basename(output_image_path)))

    return redirect(request.url)

# Route to display the processed image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Main page route (index)
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(port=5001, debug=True)
