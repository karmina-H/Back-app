from flask import Flask, jsonify, send_file
from flask_cors import CORS
import os

from picture_data import data


app = Flask(__name__)
CORS(app)

# 이미지 디렉토리 경로
IMAGE_DIRECTORY = 'C:\\back-end-app\\picture'



@app.route('/data', methods=['GET'])
def get_data():
    return jsonify(data)

@app.route('/pictures/<path:filename>', methods=['GET'])
def get_picture(filename):
    return send_file(os.path.join(IMAGE_DIRECTORY, filename))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
