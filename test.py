# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 09:34:10 2019

@author: deepakc
"""

from PIL import Image
import os
import numpy as np
import sys
import csv

image_folder_path = sys.argv[1]
model_path = sys.argv[2]

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

img_width = 225
img_height = 300

def loadImageList(folderPath):
    filePathList = []
    for files in os.listdir(folderPath):
        if files.endswith(".jpg"):
            filePathList.append(os.path.join(folderPath, files))
    return filePathList

filePath = r"D:\Deepak\Text Localization\New folder\Pattern\data\images\0bdab93e-6c01-43c7-8e9a-eae5ce60952d1527484116329-Monte-Carlo-Men-Tshirts-4731527484116179-2.jpg"

def loadImage(fileList):
    X = []
    for filePath in fileList:
        img = Image.open(filePath)
        if img.size != (img_width, img_height):
            img = img.resize((img_width, img_height))
        X.append(img.getdata())
        
    X = np.asarray(X).reshape((-1, img_height, img_width, 3))/255
    
    return X
        
        

filePathlist = loadImageList(image_folder_path)

meta_filePath = os.path. join(model_path, [file for file in os.listdir(model_path) if file.endswith(".meta")][0])

predction_result = open('output.csv', mode='w')
csv_writer = csv.writer(predction_result)
csv_writer.writerow(["filename", "neck", "sleeve_length","pattern"])
    
with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph(meta_filePath)
  new_saver.restore(sess, tf.train.latest_checkpoint(model_path))
  
  graph = tf.get_default_graph()
  input_x = graph.get_tensor_by_name("input_x:0")
  output_y_0 = graph.get_tensor_by_name("output_y_0/Softmax:0")
  output_y_1 = graph.get_tensor_by_name("output_y_1/Softmax:0")
  output_y_2 = graph.get_tensor_by_name("output_y_2/Softmax:0")
  
  for i in range(0, len(filePathlist), 32):
      
      if i + 32 < len(filePathlist):
          filePathBatch = filePathlist[i:i+32]
    
      else:
          filePathBatch = filePathlist[i:]
          
      train_x = loadImage(filePathlist)
      
      pred_y_0, pred_y_1, pred_y_2 = sess.run([output_y_0, output_y_1, output_y_2], feed_dict={input_x:train_x})
      
      pred_y_0 = np.argmax(pred_y_0, axis = 1)
      pred_y_1 = np.argmax(pred_y_1, axis = 1)
      pred_y_2 = np.argmax(pred_y_2, axis = 1)
      
      for i in range(len(filePathBatch)):
          csv_writer.writerow([filePathBatch[i].split("\\")[-1], pred_y_0[i], pred_y_1[i], pred_y_2[i]])
      
predction_result.close()      

"""
img_width = 225
img_height = 300
weight_decay = 0.001

image_folder_path = sys.argv[1]
csv_path = sys.argv[2]

data_csv = pd.read_csv(csv_path)
data_csv.head()
data_csv.info()
#
#for col_name in data_csv.columns[1:]:
#    print(col_name)
#    pd.value_counts(data_csv[col_name]).plot.bar()
#    plt.show()
    
oneHotEncList = []
for col_name in data_csv.columns[1:]:
    series = data_csv[col_name]
    series_notna_index = series.notna()
    series_notna = series[series_notna_index]
    series_notna = np.expand_dims(np.asarray(series_notna), axis = 1)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(series_notna)
    oneHotEncList.append(enc)
        

    

#from DataAugmentation import DataAugmentation


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
DAobj = DataAugmentation()

def loadImage(data_csv, b_s = None):
    if b_s is not None:
        data_csv = data_csv.sample(b_s)
        
    X = []
    Y_indice = []
    for i in range(data_csv.shape[0]):
        fileName = data_csv.iloc[i]["filename"]
        filePath = os.path.join(image_folder_path, fileName)
        if not os.path.exists(filePath):
            continue
        Y_indice.append(i)
        img = Image.open(filePath)
        img = DAobj(img)
        X.append(img.getdata())
    
    Y = []    
    for i, col in enumerate(data_csv.columns[1:]):
       series = data_csv.iloc[Y_indice][col] 
       series = series.fillna(value = -1)
       series = np.expand_dims(np.asarray(series), axis = 1)
       Y.append(oneHotEncList[i].transform(series).toarray())
       
    X = np.asarray(X).reshape((-1, img_height, img_width, 3))/255
        
        
    return X, Y
        
data_csv = shuffle(data_csv)
data_csv_train, data_csv_test = train_test_split(data_csv, train_size  = 0.9)


class_weights_list = []

for col in data_csv_train.columns[1:]:
    series = data_csv_train[col]
    series_notna_index = series.notna()
    series_notna = series[series_notna_index]
    class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(series_notna),
                                                series_notna)
    class_weights_list.append(dict(enumerate(class_weights)))


input_x = tf.placeholder(tf.float32, shape = [None, img_height, img_width, 3])

input_y_0 = tf.placeholder(tf.float32, shape = [None, len(oneHotEncList[0].categories_[0])])
input_y_1 = tf.placeholder(tf.float32, shape = [None, len(oneHotEncList[1].categories_[0])])
input_y_2 = tf.placeholder(tf.float32, shape = [None, len(oneHotEncList[2].categories_[0])])

class_weights_0 = tf.placeholder(tf.float32, shape = [None])
class_weights_1 = tf.placeholder(tf.float32, shape = [None])
class_weights_2 = tf.placeholder(tf.float32, shape = [None])

with tf.variable_scope("InceptionModel"):
    inception_model = tf.keras.applications.InceptionV3(include_top = False, input_tensor = input_x, weights = "imagenet")
#    inception_model = tf.keras.applications.VGG16(include_top = False, input_tensor = input_x, weights = "imagenet")
    
mixed3 = tf.get_default_graph().get_tensor_by_name("InceptionModel/mixed3/concat:0")
#mixed3 = tf.get_default_graph().get_tensor_by_name("InceptionModel/block3_pool/MaxPool:0")


global_pool = tf.math.reduce_max(mixed3, axis = [1, 2])

#x = tf.layers.conv2d(mixed3, 128, 3, activation=tf.nn.relu, kernel_initializer=tf.initializers.he_normal())
#x = tf.math.reduce_max(x, axis = [1, 2])
x = tf.layers.dense(global_pool, 64, activation = tf.nn.relu, kernel_initializer=tf.initializers.he_normal())
output_y_0 = tf.layers.dense(x, len(oneHotEncList[0].categories_[0]), activation=tf.nn.softmax)

#x = tf.layers.conv2d(mixed3, 128, 3, activation=tf.nn.relu, kernel_initializer=tf.initializers.he_normal())
#x = tf.math.reduce_max(x, axis = [1, 2])
x = tf.layers.dense(global_pool, 64, activation = tf.nn.relu, kernel_initializer=tf.initializers.he_normal())
output_y_1 = tf.layers.dense(x, len(oneHotEncList[1].categories_[0]), activation=tf.nn.softmax)

#x = tf.layers.conv2d(mixed3, 128, 3, activation=tf.nn.relu, kernel_initializer=tf.initializers.he_normal())
#x = tf.math.reduce_max(x, axis = [1, 2])
x = tf.layers.dense(global_pool, 64, activation = tf.nn.relu, kernel_initializer=tf.initializers.he_normal())
output_y_2 = tf.layers.dense(x, len(oneHotEncList[2].categories_[0]), activation=tf.nn.softmax)

error = -tf.reduce_sum(input_y_0*tf.log(tf.clip_by_value(output_y_0,1e-10,1.0)), axis = -1)
scaled_error_0 = tf.multiply(error, class_weights_0)
scaled_error_0 = tf.reduce_mean(scaled_error_0)

error = -tf.reduce_sum(input_y_1*tf.log(tf.clip_by_value(output_y_1,1e-10,1.0)), axis = -1)
scaled_error_1 = tf.multiply(error, class_weights_1)
scaled_error_1 = tf.reduce_mean(scaled_error_1)

error = -tf.reduce_sum(input_y_2*tf.log(tf.clip_by_value(output_y_2,1e-10,1.0)), axis = -1)
scaled_error_2 = tf.multiply(error, class_weights_2)
scaled_error_2 = tf.reduce_mean(scaled_error_2)

vars   = tf.trainable_variables() 
lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars ]) * 0.001

cross_entropy_loss = tf.reduce_mean([scaled_error_0, scaled_error_1, scaled_error_2])
loss = cross_entropy_loss + lossL2

optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
opt = optimizer.minimize(loss)

initialize_variables_list = list(set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))^set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='InceptionModel')))
init = tf.initialize_variables(initialize_variables_list)

saver = tf.train.Saver()


sess = tf.keras.backend.get_session()

sess.run(init)

reset_optimizer_op = tf.variables_initializer(optimizer.variables())
sess.run(reset_optimizer_op)

n_iteration = 1000
batch_size = 16
#X_test, Y_test = loadImage(data_csv_test)

for step_count in range(n_iteration):
    train_x, train_y = loadImage(data_csv_train, batch_size)
    train_class_weight = []
    for i, c_w in enumerate(class_weights_list):
        classLabel = np.argmax(train_y[i], axis = 1)
        train_class_weight.append(np.vectorize(class_weights_list[i].get)(classLabel))
    
    
    _, temp, cost_train = sess.run([opt, loss, cross_entropy_loss], feed_dict={input_x:train_x, input_y_0:train_y[0], input_y_1:train_y[1], input_y_2:train_y[2], class_weights_0:train_class_weight[0], class_weights_1:train_class_weight[1], class_weights_2:train_class_weight[2]})
    
    if step_count % 50 == 0:
        test_x, test_y = loadImage(data_csv_test, 64)
        test_class_weight = []
        for i, c_w in enumerate(class_weights_list):
            classLabel = np.argmax(test_y[i], axis = 1)
            test_class_weight.append(np.vectorize(class_weights_list[i].get)(classLabel))
        
        
        temp, cost_test = sess.run([loss, cross_entropy_loss], feed_dict={input_x:test_x, input_y_0:test_y[0], input_y_1:test_y[1], input_y_2:test_y[2], class_weights_0:test_class_weight[0], class_weights_1:test_class_weight[1], class_weights_2:test_class_weight[2]})
    
    
        print("Iteration Number", step_count, "Train Loss", cost_train, "Test Loss", cost_test)

saved_path = saver.save(sess, './SavedModel')

#from sklearn.metrics import confusion_matrix
#predict_y_0 = np.empty((0,))
#predict_y_1 = np.empty((0,))
#predict_y_2 = np.empty((0,))
#
#label_y_0 = np.empty((0,))
#label_y_1 = np.empty((0,))
#label_y_2 = np.empty((0,))
#
#
#
#for i in range(0, data_csv_train.shape[0], 32):
#    print(i)
#    train_x, train_y = loadImage(data_csv_train[i:i+32])
#    temp_0, temp_1, temp_2 = sess.run([output_y_0, output_y_1, output_y_2], feed_dict={input_x:train_x})
#    
#    notnan_indice_0 = np.nonzero(np.max(train_y[0], axis = 1))[0]
#    notnan_indice_1 = np.nonzero(np.max(train_y[1], axis = 1))[0]
#    notnan_indice_2 = np.nonzero(np.max(train_y[2], axis = 1))[0]
#
#    predict_y_0 = np.concatenate((predict_y_0, np.argmax(temp_0[notnan_indice_0], axis = 1)))
#    predict_y_1 = np.concatenate((predict_y_1, np.argmax(temp_1[notnan_indice_1], axis = 1)))
#    predict_y_2 = np.concatenate((predict_y_2, np.argmax(temp_2[notnan_indice_2], axis = 1)))
#    
#    label_y_0 = np.concatenate((label_y_0, np.argmax(train_y[0][notnan_indice_0], axis = 1)))
#    label_y_1 = np.concatenate((label_y_1, np.argmax(train_y[1][notnan_indice_1], axis = 1)))
#    label_y_2 = np.concatenate((label_y_2, np.argmax(train_y[2][notnan_indice_2], axis = 1)))
#    
#
#confusion_matrix_0 = confusion_matrix(label_y_0, predict_y_0)
#confusion_matrix_1 = confusion_matrix(label_y_1, predict_y_1)
#confusion_matrix_2 = confusion_matrix(label_y_2, predict_y_2)

"""