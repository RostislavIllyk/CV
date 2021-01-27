# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 22:01:21 2021

@author: rost_
"""


# USAGE
# python simple_request.py

# import the necessary packages
import numpy as  np
import requests
from matplotlib.pyplot import imshow, imsave

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://0.0.0.0:8080/predict"
IMAGE_PATH = './test/999999.jpg'

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

img = np.array(r["pic"])
imshow(img)

filename = './tempo/test.png'
imsave(filename, img)
print('You can find your result at "./tempo/test.png" ')