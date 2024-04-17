from collections import defaultdict
from email.policy import default
import tensorflow as tf
import numpy as np
import random
from math import floor
from copy import copy, deepcopy
import matplotlib.pyplot as plt
from operator import itemgetter
import scipy.io
import codecs
import os
import os.path
import shutil
import string
import sys
from PIL import Image
import warnings
import gzip
import pandas as pd

random.seed(100)

class FMNIST:
    def __init__(self,path,kind='train'):
        self.path=path
        self.kind=kind
        self.images,self.targets= self.load_fmnist()
        self.datasets = pd.DataFrame(data={"images":list(self.images),'targets':(self.targets)})
        # self.train_tensor = tf.data.Dataset.from_tensor_slices((self.images, self.targets))
        # self.train=tf.data.Dataset.zip((self.images,self.targets))
        # print(self.targets.cardinality().numpy())
    
    def __len__(self):
        return len(self.targets)
    
    def my_func(self,img):
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        return img

    def load_fmnist(self,reshape_dim=(28,28,1)):
        labels_path = os.path.join(self.path,'%s-labels-idx1-ubyte.gz' % self.kind)
        images_path = os.path.join(self.path,'%s-images-idx3-ubyte.gz' % self.kind)
        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                offset=16).reshape(len(labels),*reshape_dim)
        
        return np.multiply(images.astype(np.float32), 1.0 / 255.0), labels#@self.my_func(images),tf.convert_to_tensor(labels,dtype=tf.int32)
        
        # np.multiply(images.astype(np.float32), 1.0 / 255.0), labels
        # return tf.data.Dataset.from_tensor_slices(images), tf.data.Dataset.from_tensor_slices(labels)


class EMNIST:
    def __init__(self,path,name='digits',kind='train'):
        self.path=path
        self.kind=kind
        self.name=name
        self.images,self.targets= self.load_EMNIST()
        self.datasets = pd.DataFrame(data={"images":list(self.images),'targets':(self.targets)})
        # self.train_tensor = tf.data.Dataset.from_tensor_slices((self.images, self.targets))
        # self.train=tf.data.Dataset.zip((self.images,self.targets))
        # print(self.targets.cardinality().numpy())
    
    def __len__(self):
        return len(self.targets)
        

    def load_EMNIST(self,reshape_dim=(28,28,1)):
        labels_path = os.path.join(self.path,'emnist-'+self.name+'-%s-labels-idx1-ubyte.gz' % self.kind)
        images_path = os.path.join(self.path,'emnist-'+self.name+'-%s-images-idx3-ubyte.gz' % self.kind)
        # print(labels_path)
        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
        with gzip.open(images_path, 'rb') as imgpath:
            # print(imgpath.read())
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                offset=16).reshape(len(labels),*reshape_dim)
            # print(images)
            # print(len(labels))
        return np.rot90(np.flip(np.multiply(images.astype(np.float32), 1.0 / 255.0),axis=1), axes=(2,1)), labels


class MNIST:
    def __init__(self, path, kind='train'):
        self.path = path
        self.kind = kind
        self.images, self.targets = self.load_mnist()
        self.datasets = pd.DataFrame(data={"images":list(self.images), 'targets':(self.targets)})

    
    def __len__(self):
        return len(self.targets)
    
    def my_func(self, img):
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        return img

    
    def load_mnist(self, reshape_dim=(28,28,1)):
        labels_path = os.path.join(self.path,'%s-labels-idx1-ubyte.gz' % self.kind)
        images_path = os.path.join(self.path,'%s-images-idx3-ubyte.gz' % self.kind)
        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                offset=16).reshape(len(labels),*reshape_dim)
        
        return np.multiply(images.astype(np.float32), 1.0 / 255.0), labels#@self.my_func(images),tf.convert_to_tensor(labels,dtype=tf.int32)
        

class GTSRB:
    def __init__(self, path, kind='train'):
        self.path = path
        self.kind = kind
        self.images, self.targets = self.direct_numpy_import() 
        #self.load_GTSRB()
        self.datasets = pd.DataFrame(data={"images":list(self.images), 'targets':(self.targets)})

    
    def __len__(self):
        return len(self.targets)
    
    def my_func(self, img):
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        return img

    
    def load_GTSRB(self, reshape_dim=(32,32)):
        x = []
        y = []
        no_of_classes = 43
        print("number of classes in GTSRB data:", no_of_classes)
        # read the test.csv file; this file contains all the labels
        test_csv_data = pd.read_csv(os.path.join(self.path,"Test.csv"))
        class_ids = np.array(test_csv_data['ClassId'])
        if self.kind=='test':
            # read all images from test folder
            local_path = os.path.join(self.path, 'Test')
            print("path of test images: {}".format(local_path))
            # load all images from folder
            images = os.listdir(local_path)
            print("total test images {}".format(len(images)))
            for i in range(len(images)):
                image = Image.open(local_path+ '\\'+ images[i])
                image = image.resize(reshape_dim)
                image = np.array(image)
                x.append(image)
                y.append(class_ids[i])
        else:
            for i in range(no_of_classes):
                # print("entering the training....")
                local_path = os.path.join(self.path,'Train',str(i))

                # print("Path:", local_path)
                
                # get the all images from this folder
                images = os.listdir(local_path)

                print("Length of images in class-"+str(i)+":", len(images))
                for img in images:
                    try:
                        image = Image.open(local_path + '//'+ img)
                        image = image.resize(reshape_dim)
                        image = np.array(image)
                        #sim = Image.fromarray(image)
                        x.append(image)
                        y.append(i)
                    except:
                        print("Error loading image")            

        """ convert into numpy arrays"""
        x = np.array(x)
        y = np.array(y)
        """ verify shape """
        print("{}, {}".format(x.shape, y.shape))
        return np.multiply(x.astype(np.float32), 1.0 / 255.0), y

    def direct_numpy_import(self):
        # os.path.join(self.path,"test_labels.npy")
        if self.kind=='test':
            X_test = np.load(os.path.join(self.path,"test_data.npy")).astype("float32")
            y_test = np.load(os.path.join(self.path,"test_labels.npy")).astype("float32")
            print("X_test=", X_test.shape)
            print("Y_test=", y_test.shape)
            return np.multiply(X_test, 1.0 / 255.0),y_test
        else:
            x = np.load(os.path.join(self.path,"data.npy")).astype("float32")
            y = np.load( os.path.join(self.path,"labels.npy")).astype("float32")
            print("X_train=", x.shape)
            print("y_train=", y.shape) 
            return np.multiply(x, 1.0 / 255.0),y
        

        

    











if __name__ == "__main__":
    path='.//data//EMNIST//gzip'
    Datasets=EMNIST(path)
    # print(Datasets.targets)
    # print(Datasets.images)

# if __name__ == "__main__":
#     path='.//Data//data//FashionMNIST//raw'
#     Datasets=FMNIST(path)
#     # print(Datasets.datasets.targets.shape)
#     # print(list(Datasets.targets))