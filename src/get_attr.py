from keras.preprocessing.image import load_img , img_to_array
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import numpy as np
import glob
import os.path

#---------- set gpu using tf ---------------------------
def get_session(gpu_fraction=0.5):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(get_session())

IMG_WIDTH = 178
IMG_HEIGHT = 218


class InceptionV3Model:
    def __init__(self):
        model = load_model('InceptionV3_models/keras_celebA_multilabel_trained_model.h5')
        model._make_predict_function()
        self.model = model

    def load_reshape_img(self, path) :
        img = load_img(path)
        img = img_to_array(img)/255.
        img = img.reshape((1,) + img.shape)

        return img

    def attr_classification(self, pre) :
        attrdict = {1 : "Male", 2 : "Normal", 3 : "No", 4 : "No", 5 : "Normal"}
        attrdict[1] = 'Female' if pre[20] < 0.1 else 'Male'
        attrdict[2] = 'Ugly' if pre[2] < 0.1 else 'Attractive' if pre[2] > 0.8 else 'Normal'
        attrdict[3] = 'Yes' if pre[24] < 0.8 else 'No'
        attrdict[4] = 'Smiling' if pre[31] > 0.6 else 'No'
        attrdict[5] = 'Young' if pre[39] > 0.87 else 'Old' if pre[39] < 0.7 else 'Normal'
        return attrdict

    def inference(self):
        files = sorted(glob.glob('src/imageSave/*'))

        _result = []

        for path in files :
            img = self.load_reshape_img(path)
            result = self.model.predict(img)
            prediction = result[0]
            attr = list(self.attr_classification(prediction).values())
            _result.append(attr)
        # _result = ['Attractive','gender','Beard','Smile','Young'] //dummy
        return _result


