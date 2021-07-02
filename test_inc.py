from tensorflow.keras import Input
from tensorflow.keras.layers import *
from tensorflow.keras.applications import *
from tensorflow.keras import *
import tensorflow as tf
tf.keras.backend.set_image_data_format('channels_last')
def conv_block(inputs, num_filter):
  # inputs = Input(shape = (256,256,30,1))

  tower0 = Conv3D(num_filter, kernel_size=(1,1,1),strides = (1,1,1),padding = "same")(inputs)
  # print(tower0.shape)

  tower1 = Conv3D(num_filter, kernel_size=(1,1,1),strides = (1,1,1),padding = "same")(inputs)
  tower1 = Conv3D(num_filter, kernel_size=(3,3,3),strides = (1,1,1),padding = "same")(tower1)
  # print(tower1.shape)

  tower2 = MaxPool3D(pool_size=(3,3,3),padding = "same" )(inputs)
  tower2 = Conv3D(num_filter, kernel_size=(1,1,1),strides = (1,1,1),padding = "same")(inputs)
  # print(tower2.shape)

  tower3 = Conv3D(num_filter, kernel_size = (1,1,1),strides = (1,1,1),padding = "same")(inputs)
  tower3 = Conv3D(num_filter, kernel_size = (5,5,5),strides = (1,1,1),padding = "same")(tower3)
  # print(tower3.shape)

  merge = tf.concat([tower0,tower1,tower2,tower3],axis = 3)
  return merge
# print(merge.shape)
def build_model():
  inputs = Input(shape=(256,256,30,1))
  x = inputs 

  x1 = conv_block(x,128)
  x2 = conv_block(x1,64)
  x3 = conv_block(x2,32)

  x4 = Conv3D(32,kernel_size = (1,1,1),strides= (1,1,1),padding = "same")(x3)
  out = layers.GlobalAveragePooling3D()(x4)
  out = layers.Dense(units=512, activation="relu")(out)
  out = layers.Dropout(0.3)(out)

  outputs = layers.Dense(units=1, activation="sigmoid")(out)
  model = tf.keras.Model(inputs,outputs)
  return model
model = build_model()