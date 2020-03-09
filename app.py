

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import cv2


import argparse
import os
import re
import sys
import tarfile
from shutil import copy

from flask import Flask,render_template, request,redirect, url_for,jsonify
from werkzeug import secure_filename
import _thread
from threading import Thread

import numpy as np
import pickle


from six.moves import urllib


import string
import random
from numpy import genfromtxt

from main import compute
from hist import GetTransformedImage
from feature_extraction import extract_features

FLAGS = None

app =Flask(__name__)

PathForUploads=""

PathForOutputs=""

OUTPUT_FOLDER='static'+'/'+'img'+'/'+'output_Dir'

UPLOAD_FOLDER='uploads'


EmotionalDist=[]

app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg', 'gif'])

target_hlfeat = genfromtxt('input_Dir/' + 'emo6_feat.csv', delimiter=',')


def main(_):
    global tmpimage
    image = (FLAGS.image_file if FLAGS.image_file else
    os.path.join(FLAGS.model_dir, tmpimage))


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def calculate_features_for_target_image(image):

    extract_features(UPLOAD_FOLDER,image)

    return 1





@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')



@app.route('/upload',methods=['POST'])
def UploadFile():
    global tmpimage

    if 'file' not in request.files:
        return jsonify({"success":False})
    File = request.files['file']
    if File.filename=='':
        flash('No selected file')
        return jsonify({"success":False})

    if File and allowed_file(File.filename):
        Extention =File.filename.rsplit('.', 1)[1].lower()
        name=id_generator()+'.'+Extention
        File.filename=name
        PathForUploads=os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(File.filename))
        tmpimage=name
        File.save(PathForUploads)
        return jsonify({"success":True})

    else:
        return jsonify({"success":False})





@app.route('/my_endpoint', methods=['GET','POST'])
def my_endpoint_handler():

    global PathForOutputs
    global EmotionalDist

    if request.is_json:
        my_new_object = request.get_json()
        EmotionalDist=my_new_object["x"]
        print(EmotionalDist)


    def handle_sub_view():
        global PathForUploads
        global tmpimage
        global EmotionalDist

        with app.test_request_context():

            calculate_features_for_target_image(tmpimage)
            des_images,accumu_dist,works=compute(tmpimage,EmotionalDist,target_hlfeat)
            image = cv2.imread(UPLOAD_FOLDER+"/"+tmpimage)

            if des_images!=None:
                GetTransformedImage(image, des_images, accumu_dist, OUTPUT_FOLDER+"/"+tmpimage)

            else:

                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(OUTPUT_FOLDER+"/"+tmpimage,gray_image)

            # Do the processing here
    thread=Thread(target=handle_sub_view)
    thread.start()
    thread.join()
    return jsonify(OUTPUT_FOLDER+"/"+tmpimage)



if __name__ == '__main__':
    app.run(threaded=True)
