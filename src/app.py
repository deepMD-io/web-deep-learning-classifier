
import yaml
import sys
from io import BytesIO
from typing import List, Dict, Union, ByteString, Any
import os
import flask
from flask import Flask
import requests
import json
import numpy as np
import base64

from keras.preprocessing import image

with open("src/config.yaml", 'r') as stream:
    APP_CONFIG = yaml.full_load(stream)

app = Flask(__name__)





def load_image_url(url: str) -> Image:
    response = requests.get(url)
    img = open_image(BytesIO(response.content))
    return img


def load_image_bytes(raw_bytes: ByteString) -> Image:
    img = open_image(BytesIO(raw_bytes))
    return img


def predict(imgg, n: int = 3) -> Dict[str, Union[str, List]]:
    mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
    inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}

    img_width, img_height = 224, 224

    img = image.load_img(imgg, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x.astype('float32') / 255.0
    data = json.dumps({"signature_name": "classification",
                       "instances": x.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://35.222.46.150:8501/v1/models/covid:predict',
                                  data, headers=headers)

    print(inv_mapping[predictions.argmax(axis=1)[0]])
    # print(predictions[0][0], predictions[0][1], predictions[0][2])
    predictions = numpy.array(json.loads(json_response.text)["predictions"])
    # print('Prediction: {}'.format(inv_mapping[predictions.argmax(axis=1)[0]]))
    print('Confidence')
    out = 'Normal: {:.3f}, Pneumonia: {:.3f}, COVID: {:.3f}'.format(predictions[0][0]*100, predictions[0][1]*100, predictions[0][2]*100)
    print(out)
    #return {"class": str(pred_class), "predictions": predictions}
    return {out}


@app.route('/api/classify', methods=['POST', 'GET'])
def upload_file():
    if flask.request.method == 'GET':
        url = flask.request.args.get("url")
        img = load_image_url(url)
    else:
        bytes = flask.request.files['file'].read()
        img = load_image_bytes(bytes)
    res = predict(img)
    return flask.jsonify(res)




@app.route('/ping', methods=['GET'])
def ping():
    return "pong"


@app.route('/config')
def config():
    return flask.jsonify(APP_CONFIG)


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"

    response.cache_control.max_age = 0
    return response


@app.route('/<path:path>')
def static_file(path):
    if ".js" in path or ".css" in path:
        return app.send_static_file(path)
    else:
        return app.send_static_file('index.html')


@app.route('/')
def root():
    return app.send_static_file('index.html')


def before_request():
    app.jinja_env.cache = {}


model = load_model('models')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)

    if "prepare" not in sys.argv:
        app.jinja_env.auto_reload = True
        app.config['TEMPLATES_AUTO_RELOAD'] = True
        app.run(debug=False, host='0.0.0.0', port=port)
        # app.run(host='0.0.0.0', port=port)
