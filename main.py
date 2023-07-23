from skimage.util.shape import view_as_windows

import cv2
import os

import numpy as np
import math

labels = ['positive', 'negative']

def get_data(data_dir):
    data = [] 
    for label in labels: 
        class_num = labels.index(label)
        for img in os.listdir(data_dir):
            try:
                img_arr = cv2.imread(os.path.join(data_dir, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (512, 384)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
        
    data = np.array(data)
    print(data)
    return data

negative = get_data('C:\\Users\\capoo\\Documents\\GitHub\\Internal Assessments\\Math\\Malaria\\negative')
positive = get_data('C:\\Users\\capoo\\Documents\\GitHub\\Internal Assessments\\Math\\Malaria\\positive')

x_train = negative[0:int(0.8 * len(negative))]
x_train = np.append(x_train, positive[0:int(0.8 * len(positive))])

y_train = []
for i in range(int(0.8 * len(negative))):
    y_train.append(0)

for i in range(int(0.8 * len(positive))):
    y_train.append(1)

x_val = negative[int(0.8 * len(negative)):]
x_val = np.append(x_val, positive[0:int(0.8 * len(positive))])

y_val = []
for i in range(int(0.2 * len(negative))):
    y_train.append(0)

for i in range(int(0.2 * len(positive))):
    y_train.append(1)


# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, 512, 384, 1)
y_train = np.array(y_train)

x_val.reshape(-1, 512, 384, 1)
y_val = np.array(y_val)

def error(predicted, actual):
    return 0.5 * (predicted - actual)**2

def sigmoid(x):
    return 1 / (1 + Math.exp(-x))

def dense(x, predicted):
    w = np.random.rand(x.shape[1], 2) #features x outputs 
    for i in range(100):
        f = np.dot(x, w)
        w += (sigmoid(f) - predicted) * np.dot(sigmoid(f), (1 - sigmoid(f))) * f
    
    return np.dot(f, w)

def MaxPooling(x): #max pooling with 2x2 filter and stride 2 -- only works for dim % 4 = 0
    pooled = np.empty((int((x.shape[0]) / 2), int((x.shape[1]) / 2)))

    for row in range(0, int(x.shape[0]), 2):
        for column in range(0, int(x.shape[1]), 2):
            max = x[row, column]
            if (x[row, column + 1] > max):
                max = x[row, column + 1]
            if (x[row + 1, column] > max):
                max = x[row + 1, column]
            if (x[row + 1, column + 1] > max):
                max = x[row + 1, column + 1]

            pooled[math.ceil(row / 2), math.ceil(column / 2)] = max
    return pooled

def Convolution(x):
    kernel = []

    output_shape = (x.shape[0] - kernel.shape[0]) + 1
    convolved_matrix = view_as_windows(x, kernel.shape).reshape(output_shape*output_shape, kernel.shape[0]*2)
    convolved_matrix = np.dot(kernel.flatten(), convolved_matrix)
 
    return convolved_matrix.reshape(output_shape, output_shape) 