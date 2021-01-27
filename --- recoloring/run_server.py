# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 23:14:26 2020

@author: rost_
"""

import numpy as np # linear algebra

from matplotlib.pyplot import imshow, imsave
from skimage.color import lab2rgb, rgb2lab


from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Dense, Dropout, RepeatVector, Reshape, concatenate
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3

from skimage.transform import resize
import os

from PIL import Image
import flask
from flask import request, render_template, redirect, url_for

#from flask import send_file
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
UPLOAD_FOLDER = './static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


global feature_extract_model
global model
global data


class feature_extract_model():
    
    def __init__(self):
        input_tensor = Input(shape=(224, 224, 3))
        self.model_f = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
        
    def predict(self, x):
        return self.model_f.predict(x)




def prepare_image(image, feature_extract_model):
    
    image = img_to_array(image) / 255
    width, height, depth = image.shape
    original_size = [width, height]
    print(width, height, depth)
    image=resize(image, (224, 224))
    lab_image = rgb2lab(image)            
    lab_image_norm = (lab_image + [0, 128, 128]) / [100, 255, 255]
    
    # The input will be the black and white layer
    X = lab_image_norm[:,:,0]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    bandw = lab_image_norm.copy()
    bandw[:,:,1]=0
    bandw[:,:,2]=0
                
    bandw =  bandw.reshape(1, bandw.shape[0], bandw.shape[1], 3)
    fc2_features = feature_extract_model.predict(bandw)
    fc2_features = fc2_features.flatten()
    
    img_features = fc2_features.astype('float32')
    img = X.astype('float32')
    img =  img.reshape(1, img.shape[0], img.shape[1], 1)
    img_features =  img_features.reshape(1, img_features.shape[0])

    return img, img_features, original_size





def def_and_load_model():
    
    feature_size = 25088
    
    # feature extractor model
    inputs1 = Input(shape=(feature_size,))
    image_feature = Dropout(0.5)(inputs1)
    image_feature = Dense(1024, activation='relu')(image_feature)
    
    img_h = 224
    img_w = 224
    
    inputs2 = Input(shape=(img_h, img_w, 1,))
    
    encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(inputs2)
    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
    encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
    
    
    concat_shape = (np.uint32(encoder_output.shape[1]), np.uint32(encoder_output.shape[2]),np.uint32(inputs1.shape[-1]))
    
    image_feature = RepeatVector(concat_shape[0]*concat_shape[1])(inputs1)
    image_feature = Reshape(concat_shape)(image_feature)
    
    
    fusion_output = concatenate([encoder_output, image_feature], axis=3)
        
    
    decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
    
    #decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    
    
    model = Model(inputs=[inputs1, inputs2], outputs=decoder_output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
    
    
    
    
    try:
        print('try')
        model = load_model('models/model_9_224_224_imp_6491.h5')
        print('model weights are loaded...')
    except:
        print('not loaded !!!')
    
    
    return model
    



@app.route("/predict", methods=["POST"])
def predict():
    
    data = {"success": False, "pic": None}
    
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
			# read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))    
            
            if image.mode != "RGB":
                    image = image.convert("RGB")
                    
            data["success"] = True        

    img, img_features, size = prepare_image(image, feature_extract_model)
    
    output = model.predict([img, img_features])
    
    
    cur = np.zeros((224, 224, 3))
    cur[:,:,0] = img[0][:,:,0]
    cur[:,:,1:] = output[0]
    
    
    cur = (cur * [100, 255, 255]) - [0, 128, 128]
    rgb_image = lab2rgb(cur)
    
    ratio = size[0]/size[1]
    
    w=300
    h=int(w/ratio)
    rgb_image=resize(rgb_image, (w, h))

    data["pic"] = rgb_image.tolist()        

#    imshow(rgb_image)

    return flask.jsonify(data)







@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
        path = os.path.join(app.config['UPLOAD_FOLDER'], 'test.jpg')
        file1.save(path)

        image = load_img('./static/test.jpg')
        img, img_features, size = prepare_image(image, feature_extract_model)
        output = model.predict([img, img_features])
        
        cur = np.zeros((224, 224, 3))
        cur[:,:,0] = img[0][:,:,0]
        cur[:,:,1:] = output[0]
        cur = (cur * [100, 255, 255]) - [0, 128, 128]
        rgb_image = lab2rgb(cur)
        ratio = size[0]/size[1]
        w=300
        h=int(w/ratio)
        rgb_image=resize(rgb_image, (w, h))
    
        filename = './static/test.jpg'
        imsave(filename, rgb_image)
        
        return redirect(url_for('static', filename='test.jpg'))
    return render_template("start.html", name = "file1")






if __name__ == "__main__":
    
    
    feature_extract_model = feature_extract_model()
    model =  def_and_load_model()
    
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    app.run(host="0.0.0.0", port=8080, debug=True)











