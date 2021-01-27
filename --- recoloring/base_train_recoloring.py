# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 23:14:26 2020

@author: rost_
"""

import numpy as np 

from matplotlib.pyplot import imshow, imsave
from skimage.color import lab2rgb, rgb2lab


from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Dense, Dropout, RepeatVector, Reshape, concatenate
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
import random

from skimage.transform import resize
import os



class feature_extract_model():
    
    def __init__(self):
        input_tensor = Input(shape=(224, 224, 3))
        self.model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
        
    def predict(self, x):
        return self.model.predict(x)





def get_inf(path, feature_extract_model):
    
    set_of_train = 35000

    
    count=0
    X_all = []
    y_all = []
    original_size_list=[]
    features=[]
    d_list=[]
    
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            
            INPUT_IMAGE_SRC = os.path.join(dirname, filename)
            d_list.append(INPUT_IMAGE_SRC)
    
    if len(d_list) < set_of_train:
        list_to_run = d_list
        
    else:
        list_to_run = random.sample(d_list, set_of_train)
    
    
    for INPUT_IMAGE_SRC in list_to_run:

            image = img_to_array(load_img(INPUT_IMAGE_SRC)) / 255
            width, height, depth = image.shape
            original_size_list.append([width, height])
            print(width, height, depth, count)
            image=resize(image, (224, 224))
            lab_image = rgb2lab(image)            
            lab_image_norm = (lab_image + [0, 128, 128]) / [100, 255, 255]
            
            # The input will be the black and white layer
            X = lab_image_norm[:,:,0]
            # The outpts will be the ab channels
            Y = lab_image_norm[:,:,1:]
                        
            X = X.reshape(X.shape[0], X.shape[1], 1)
            Y = Y.reshape(Y.shape[0], Y.shape[1], 2)
            
            bandw = lab_image_norm.copy()
            bandw[:,:,1]=0
            bandw[:,:,2]=0
                        
            bandw =  bandw.reshape(1, bandw.shape[0], bandw.shape[1], 3)
            fc2_features = feature_extract_model.predict(bandw)
            fc2_features = fc2_features.flatten()
            
            
            fc2_features = fc2_features.astype('float32')
            X = X.astype('float32')
            Y = Y.astype('float32')
            
            
            features.append(fc2_features)
            X_all.append(X)
            y_all.append(Y)

            count = count+1

    
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    features = np.array(features)
    original_size_array = np.array(original_size_list)

    return X_all, y_all, features, original_size_array



def prepare_image(image, feature_extract_model):    
    
    image = img_to_array(image) / 255

    width, height, depth = image.shape
    original_size = [width, height]
    print(width, height, depth)
    image=resize(image, (224, 224))
    lab_image = rgb2lab(image)            
    lab_image_norm = (lab_image + [0, 128, 128]) / [100, 255, 255]
    
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
        model = load_model('models/model_9_224_224_imp_36111.h5')
        print('model weights are loaded...')
    except:
        print('not loaded !!!')
    
    
    return model
    




def train(feature_extract_model , model):

    X_train, y_train, features_train, _  = get_inf('./train_', feature_extract_model)
    #X_train, y_train, features_train, _  = get_inf('./global_train_set/places365_standard/train', feature_extract_model)
    X_test, y_test, features_test, original_size_array_test      = get_inf('./test', feature_extract_model)
    
    
    
    
    model.fit(x=[X_train, features_train] , y=y_train, batch_size=150, epochs=13, verbose=1)
    model.evaluate([X_test, features_test], y_test, batch_size=1)
    
    
    
    output = model.predict([X_test, features_test])
    
    for i in range(len(X_test)):
    
        cur = np.zeros((224, 224, 3))
        cur[:,:,0] = X_test[i][:,:,0]
        cur[:,:,1:] = output[i]
        
        
        cur = (cur * [100, 255, 255]) - [0, 128, 128]
        rgb_image = lab2rgb(cur)
        
        
        ratio = original_size_array_test[i,0]/original_size_array_test[i,1]
        
        w=300
        h=int(w/ratio)
        rgb_image=resize(rgb_image, (w, h))
        
        
        imshow(rgb_image)
        filename = './tempo/'+str(i)+'.png'
        imsave(filename, rgb_image)
    
    model.save('models/model_9_224_224_imp_37000.h5')





def predict(image, feature_extract_model , model):

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


    imshow(rgb_image)







flag_train=1


feature_extract_model = feature_extract_model()
model =  def_and_load_model()


if flag_train:
    train(feature_extract_model , model)


else:
    image = load_img('./test/IMG_0017.jpg')

    predict(image, feature_extract_model , model)














