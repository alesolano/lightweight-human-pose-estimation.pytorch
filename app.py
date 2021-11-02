from flask import Flask, jsonify

from run import Run

app = Flask(__name__)

run = Run()

@app.route('/predict', methods=['GET'])
def predict():
    boxes = run.get_bboxes_from_video('/workspace/videos/video.mp4')
    return jsonify(boxes)


if __name__ == '__main__':
    app.run(port=5001, host='0.0.0.0')