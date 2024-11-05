from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
import io
from ID_Detection import detect_id_card

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    if 'frame' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['frame']

    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    try:
        # Decode the image from the request
        img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    try:
        # Perform ID card detection
        modified_frame = detect_id_card(img)

        # Encode the modified frame back into JPEG format
        _, buffer = cv2.imencode('.jpg', modified_frame)
        img_byte_arr = io.BytesIO(buffer)

        return send_file(img_byte_arr, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)
    # app.run(host='0.0.0.0', port=5001, threaded=True)
