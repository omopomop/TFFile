#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import tensorflow as tf
import  numpy as np
#import json
#import cv2
#import os
#import matplotlib.pyplot as plt

imagefile = ''  //add imagefile here

def classify(pb_file_path):
    with tf.Graph().as_default():
        LOGDIR = 'tensorboard/'
        #open and load graph
        with open(pb_file_path, "rb") as f:
            output_graph_def = tf.GraphDef()
            output_graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(output_graph_def)
        with tf.Session() as sess:
            graph = sess.graph_def
            op = sess.graph.get_operations()
            opnames = []
            pastInput = False;

            init = tf.global_variables_initializer()
            sess.run(init)
            for key in op:
                if "input" in key.name:
                    input_x = sess.graph.get_tensor_by_name(key.name+":0")
            output = sess.graph.get_tensor_by_name(key.name+":0")
            print(np.shape(output))
            images = []

            #load and decode image
            image = tf.gfile.FastGFile(imagefile,'rb').read()
            image = tf.image.decode_bmp(image)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize_images(image,[160,160],method=0)
           # normImg = sess.run(image)
           # normImg = (normImg - 115.685692) / 59.343296
           # mul = normImg
            offset_image = tf.subtract(image.eval(), .5)
            mul_image = tf.multiply(offset_image, 2)
            mul = sess.run(mul_image)

            #print input
            coord = 0
            print("input: h={0} w={1} c={2}".format(np.size(mul,0),np.size(mul,1),np.size(mul,2)),end = '')
            for x in range (0,3):
                if x == 0:
                    coord= int(0)
                elif x == 1:
                    coord = int(np.size(mul,0)/2)
                else:
                    coord = int((np.size(mul,1)-1))
                #normImg = normImg[...,::-1]
                formatprint(mul,3,coord)
            mul = np.expand_dims(mul, axis=0)

            #add ops into list
            for item in op:
                if pastInput == True:
                    opnames.append(item.name+":0")
                if "input" in item.name:
                    pastInput = True
            names = []
            x = 0
            for x in range (0,len(opnames)):
                names.append(opnames[x])
                #Turn tensors into numpy ndarrays
                opnames[x] = sess.graph.get_tensor_by_name(opnames[x])
                opnames[x] = sess.run(opnames[x],feed_dict = {input_x: mul})
                opnames[x] = np.squeeze(opnames[x])
                coord = 0
                #print layer dump
                if opnames[x].ndim == 3:
                    print("\n\nTensorname: {0} h={1} w={2} c={3}".format(names[x],np.size(opnames[x],0), np.size(opnames[x],1),np.size(opnames[x],2)),end = '')
                    for coordinates in range (0,3):
                        if coordinates == 0:
                            coord = int(0)
                        elif coordinates == 1:
                            coord = int(np.size(opnames[x],0)/2)
                        else:
                            coord = int(np.size(opnames[x],0)-1)
                        formatprint(opnames[x],3, coord)
                elif opnames[x].ndim == 1:
                    print("\n\nTensorname: {0} c={1}".format(names[x],np.size(opnames[x],0)),end = '')
                    formatprint(opnames[x],1, 0)
            print("\n")

            #print top 5 final predictions
            print("TOP 5:")
            val = opnames[x][np.argsort(opnames[x])[-5:][::-1]]
            index = opnames[x].argsort()[-5:][::-1]
            predCount = val.size
            if val.size > 5:
                predCount = 5
            for valIndex in range (0,predCount):
                print("   feature{}".format(index[valIndex]),end=' ')
                if index[valIndex]<10:
                    print(end="  ")
                elif 10<=index[valIndex]<100:
                    print(end = ' ')
                print("Value:","{:.5f}".format(val[valIndex]))


def formatprint(opnames, ndim, coord):
    print("\nchannel data at ({0},{1}):".format(coord,coord),end = ' ') 
    for channels in range(0,np.size(opnames,(ndim-1))):
        #spacing
        if channels%16 == 0:
            if channels <10:
                print("\n  ",channels, end = '  ')
            elif 9 < channels < 100:
                print("\n ",channels, end = '  ')
            else:
                print("\n",channels, end = '  ')
        #3dim print
        if ndim == 3:
            if opnames[coord][coord][channels] >=0:
                print(end=' ')
            print("{:.5f}".format(opnames[coord][coord][channels]), end=' ')
        #1dim print
        else:
            if opnames[channels] >=0:
                print(end=' ')
            print("{:.5f}".format(opnames[channels]), end=' ')


classify('modelfile/model.pb') //add modelfile here

