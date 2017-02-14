# -*- coding: utf-8 -*-
##### change "image_dim_ordering": "tf", to  "image_dim_ordering": "th", 
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2
import numpy as np
from numpy import linalg as LA
import os
import h5py
import pickle

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    #
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    #
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    #
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    #
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    #
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    #
    if weights_path:
        model.load_weights(weights_path)

    #Remove the last two layers to get the 4096D activations
    model.layers.pop()
    model.layers.pop()
    print 'successfully loaded VGG 16'
    #
    return model

def get_imlist(path):
    """    Returns a list of filenames for
        all jpg images in a directory. """
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

def get_imglist_2(src): 
    'src下两级目录src-kindA-imgB，返回 X,Y(list),dict''注意如果文件太大，内存会爆！！！'
    folder_path = src
    folder_list = os.listdir(folder_path)
#     random.shuffle(folder_list)
#     print type (folder_list)
#     os.system('pause')
    class_list = [] ##以数字[0,1,...]来代表文件分类
    nClass = 0 # classtype: 0 1 2 3 etc. 
    class_num = len(folder_list) # 有几类，有几个目录
    img_list = []
    img_type = []
    for i in range(class_num): # different category
        new_folder_path = folder_path + '/' + folder_list[i]
        if os.path.isdir(new_folder_path): # 是否是 目录
            files = os.listdir(new_folder_path)
            for filename in files:
                img_list.append(new_folder_path+'/'+filename)
                img_type.append(folder_list[i]) # or append nClass
        nClass = nClass + 1
    return img_list,img_type

if __name__ == "__main__":

    # Test pretrained model
    model = VGG_16(r"H:\corpus_trained_model\vgg16_weights.h5") # path for vgg16_weight.h5
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    directory = './resized_data'
#     directory = r'H:\EclipseWorkspace\StatLearn\src\stat\project4\resized_data'
    imgList,img_type = get_imglist_2(directory)
    print imgList[3:8],img_type[3:8]

    feats = []
    imgNames = []

    for i, imgPath in enumerate(imgList):
#         imgPath_split = imgPath.split('/')# shape like ['.', 'resized_data', 'accordion', 'image_0001.jpg']
#         imgName = imgPath_split[2]+' '+imgPath_split[3]
#         imgNames.append(imgName)
#     print imgNames[3:8]
#     with open('imgNames.pkl','w') as f:
#         pickle.dump(imgNames,f)
#     
#     print "successfully saved all image names"
        #im = cv2.resize(  cv2.imread(imgPath), (224,224) ) #OpenCV Error: Assertion failed (ssize.area() > 0) in cv::resize, file ..\..\..\opencv-2.4.13.2\modules\imgproc\src\imgwarp.cpp, line 1968
        im = cv2.imread(imgPath)
        im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)
        out = model.predict(im)
        imgName = os.path.split(imgPath)[1]  # ['./resized_data/accordion' 'image_0001.jpg']
        normOut = out[0]/LA.norm(out[0])
        feats.append(normOut)
        imgNames.append(imgName)
        print "image %d feature extraction, total %d images" %(i, len(imgList))
   
    feats = np.array(feats)
    h5f = h5py.File('featsCNN.h5', 'w')
    h5f.create_dataset('dataset_1', data = feats)
    h5f.create_dataset('dataset_2', data = imgNames)
    h5f.close()
