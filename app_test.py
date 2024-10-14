from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index_test.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Get the JSON data from the POST request
        data = request.json

        # Decode the base64 image data to get the video frame
        image_data = base64.b64decode(data['frame'])
        np_image = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        # Draw a rectangle on the frame using OpenCV
        start_point = (100, 100)  # Top-left corner of the rectangle
        end_point = (300, 300)    # Bottom-right corner of the rectangle
        color = (0, 255, 0)       # Green color in BGR format
        thickness = 2             # Thickness of the rectangle
        cv2.rectangle(frame, start_point, end_point, color, thickness)

        # Encode the processed frame back to base64
        _, buffer = cv2.imencode('.jpg', frame)
        processed_frame = base64.b64encode(buffer).decode('utf-8')

        # Return the processed frame as JSON
        return jsonify({'processed_frame': processed_frame})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
