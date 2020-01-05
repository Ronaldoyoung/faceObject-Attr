from flask import Flask, request, send_file, g
from flask_cors import CORS
from src import InceptionV3Model, detect_faces
import numpy as np
import base64
import extract_face
import time
import json
import os
import glob

app = Flask(__name__)
CORS(app)

model = InceptionV3Model()

@app.route('/faceAttr', methods=['POST'])
def faceAttr_api():
  try :
    json_dict = request.json
    image = json_dict['imgstring']

    bounding_boxes = extract_face.slice_Image(image)
    result_attr = model.inference()

    sliceImagefiles = sorted(glob.glob('src/imageSave/*'))

    for path in sliceImagefiles :
      os.remove(path)

    bounding_boxes = bounding_boxes.tolist()
    result_json = {
      'time' : json_dict['time'],
      'face_attr' : result_attr,
      'boxes' : eval(str(bounding_boxes))
    }

    json_string = json.dumps(result_json)
    return json_string
  except KeyError:
      # image field not provided
      return 'bad request', 400
      # expose server error


@app.before_request
def set_start_time():
  g.request_start_time = time.time()


@app.after_request
def log_response_time(res):
  delta_t = time.time() - g.request_start_time
  print(f'Request completed in {delta_t}s')
  return res


if __name__ == "__main__":
  app.run(host='0.0.0.0', port=8888, debug = True)