from flask import Flask, request, jsonify
import base64
from PIL import Image
from io import BytesIO
from main import draw_landmarks
import av
import cv2

app = Flask(__name__)

@app.route('/api/upload', methods=['POST', 'GET'])
def upload():
    data = request.json
    img_data = data['image']
    img_data = base64.b64decode(img_data.split(',')[1]) # Remove 'data:image/jpeg;base64,' from the start of the string, then decode
    image = Image.open(BytesIO(img_data))

    return image

if __name__ == '__main__':
    app.run(debug=True, port=5000)
