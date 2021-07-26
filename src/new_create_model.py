import os
import glob
import time
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('INFO')
from tensorflow import keras
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model
import warnings
import argparse
import random
import logging
import pathlib
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

# surpass tf FutureWarning
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# for QAT
import tensorflow_model_optimization as tfmot
quantize_model = tfmot.quantization.keras.quantize_model

# == test input vector
A = np.ndarray(shape=(2), dtype=float)
A[0] = 51 
A[1] = 40 
# ======================
from tensorflow import pad
from tensorflow.keras.layers import Layer
class ReplicationPadding2D(Layer):
  def __init__(self, padding=(1, 1), **kwargs):
    self.padding = tuple(padding)
    super(ReplicationPadding2D, self).__init__(**kwargs)
  def compute_output_shape(self, input_shape): 
    return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])
  def call(self, input_tensor, mask=None):
    padding_width, padding_height = self.padding
    return pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0]], 'SYMMETRIC')

# create model
def build_keras_model(weights, paras, model_name):
  if model_name == "mv_model" or model_name == "mm_model":
    (input_length, output_length) = paras
    #with open("./input_vector.txt", "rb") as f:
    #  a = np.fromfile(f, dtype=np.int32)
    #  print(a.shape)
    #v = np.random.randint(0, 256, size=(4096, 1)) 
    #ans = np.dot(weights, v).astype('float32')    

#    np.save("./output_vector_float.out", ans[0])
    #ans.tofile("./output_vector_float.out", sep="", format="%s")
    #print("ans[0]:")
    #print(ans[0])
    #print("ans")
    #print(ans)
    return keras.Sequential([
      keras.layers.Flatten(input_shape=(input_length,)),
      keras.layers.Dense(output_length, weights=[weights], use_bias=False, trainable=False),
    ])    
  elif model_name == "push_model":
    input_length = paras
    input1 = keras.layers.Input(shape=(input_length*12,))
    split  = Lambda(lambda x: tf.split(x, num_or_size_splits=12, axis=1))(input1)
    vx = split[0]
    vy = split[1]
    vz = split[2]
    mag_x = split[3]
    mag_y = split[4]
    mag_z = split[5]
    ele_x = split[6]
    ele_y = split[7]
    ele_z = split[8]
    abs0  = split[9]
    abs1  = split[10]
    abs2  = split[11]
    #x = split[12]
    #y = split[13]
    #z = split[14]
    #dts = split[15]
    vx     = keras.layers.Add()([vx, mag_x])  # (1)
    vy     = keras.layers.Add()([vy, mag_y])  # (2)
    vz     = keras.layers.Add()([vz, mag_z])  # (3)
    # cross product
    v0     = keras.layers.multiply([vy, ele_z]) # (2) -> (4)
    tmp    = keras.layers.multiply([vz, ele_y]) # (3) -> (5)
    v0     = keras.layers.Subtract()([v0, tmp]) # (4) -> (6)
    v0     = keras.layers.Add()([vx, v0])       # (1) -> (7)
  
    v1     = keras.layers.multiply([vz, ele_x]) # (3) -> (8)
    tmp    = keras.layers.multiply([vx, ele_z]) # (1) -> (9)
    v1     = keras.layers.Subtract()([v1, tmp]) # (8) -> (10)
    v1     = keras.layers.Add()([vy, v1])       # (2) -> (11)

    v2     = keras.layers.multiply([vx, ele_y]) # (1) -> (12)
    tmp    = keras.layers.multiply([vy, ele_x]) # (2) -> (13)
    v2     = keras.layers.Subtract()([v2, tmp]) # (12)-> (14)
    v2     = keras.layers.Add()([vz, v2])       # (14)-> (15)
    # another cross product
    tmp    = keras.layers.multiply([v1, abs2])  # (11)-> (16)
    tmp2   = keras.layers.multiply([v2, abs1])
    tmp    = keras.layers.Subtract()([tmp, tmp2])
    vx     = keras.layers.Add()([vx, tmp])
  
    tmp    = keras.layers.multiply([v2, abs0])
    tmp2   = keras.layers.multiply([v0, abs2])
    tmp    = keras.layers.Subtract()([tmp, tmp2])
    vy     = keras.layers.Add()([vy, tmp])

    tmp    = keras.layers.multiply([v0, abs1])
    tmp2   = keras.layers.multiply([v1, abs0])
    tmp    = keras.layers.Subtract()([tmp, tmp2])
    vz     = keras.layers.Add()([vz, tmp])

    vx     = keras.layers.Add()([vx, mag_x])
    vy     = keras.layers.Add()([vy, mag_y])
    vz     = keras.layers.Add()([vz, mag_z])

    #tmp    = keras.layers.multiply([dts, vx])
    #x      = keras.layers.Add()([tmp, x])
    #tmp    = keras.layers.multiply([dts, vy])
    #y      = keras.layers.Add()([tmp, y])
    #tmp    = keras.layers.multiply([dts, vz])
    #z      = keras.layers.Add()([tmp, z])
 
    out    = keras.layers.concatenate([vx, vy, vz], axis=0)

    model  = tf.keras.models.Model(inputs=[input1], outputs=out)
    return model

  elif model_name == "imv_model":
    (input_length, output_length, ITER) = paras
    if input_length != output_length:
      print("create model: imv only support suqare weightm matrix for now, in: "+str(input_length)+", out: " +str(output_length))
      exit(0)
    print("in size: ", input_length, ", out size: ", output_length)
    #weights = np.random.randint(0, 2, size=(input_length, output_length))

    IN_W = 1
    IN_H = 1
    IN_C = input_length
    OUT_C = output_length
    F_W = S_W = 1
    F_H = S_H = 1
    
    weights = np.random.randint(1, 2, size=(F_W, F_H, IN_C, OUT_C))
    #weights = np.reshape(weights, (F_W, F_H, IN_C, OUT_C))
    print(weights)
    
    input0 = keras.layers.Input(shape=(IN_W, IN_H, IN_C))
    layer = input0
    #layer = Reshape((IN_W, IN_H, IN_C))(layer)  
    for i in range(ITER):
      #layer = keras.layers.Dense(output_length, weights=[weights], use_bias=False, trainable=False)(layer)
      layer = Conv2D(filters=OUT_C, kernel_size=(F_W, F_H), strides=(S_W, S_H), activation='linear', weights=[weights], use_bias=False, trainable=False)(layer)
    model = tf.keras.models.Model(inputs=[input0], outputs=layer)
    return model

   #if ITER != 2:
    #  print("only iter = 2 are tested for now")
    #  exit(0)
    #return keras.Sequential([
    #  keras.layers.Flatten(input_shape=(input_length,)),
    #  keras.layers.Dense(output_length, weights=[weights], use_bias=False, trainable=False),
    #  keras.layers.Dense(output_length, weights=[weights], use_bias=False, trainable=False),
    #])    
    

  elif model_name == "bmv_model":
    (input_length, output_length) = paras
    input1 = keras.layers.Input(shape=(input_length*16,))
    split  = Lambda(lambda x: tf.split(x, num_or_size_splits=16, axis=1))(input1)
    x1     = keras.layers.Dense(output_length, weights=[weights], use_bias=False, trainable=False)(split[0])
    x2     = keras.layers.Dense(output_length, weights=[weights], use_bias=False, trainable=False)(split[1])
    x3     = keras.layers.Dense(output_length, weights=[weights], use_bias=False, trainable=False)(split[2])
    x4     = keras.layers.Dense(output_length, weights=[weights], use_bias=False, trainable=False)(split[3])
    x5     = keras.layers.Dense(output_length, weights=[weights], use_bias=False, trainable=False)(split[4])
    x6     = keras.layers.Dense(output_length, weights=[weights], use_bias=False, trainable=False)(split[5])
    x7     = keras.layers.Dense(output_length, weights=[weights], use_bias=False, trainable=False)(split[6])
    x8     = keras.layers.Dense(output_length, weights=[weights], use_bias=False, trainable=False)(split[7])
    x9     = keras.layers.Dense(output_length, weights=[weights], use_bias=False, trainable=False)(split[8])
    x10    = keras.layers.Dense(output_length, weights=[weights], use_bias=False, trainable=False)(split[9])
    x11    = keras.layers.Dense(output_length, weights=[weights], use_bias=False, trainable=False)(split[10])
    x12    = keras.layers.Dense(output_length, weights=[weights], use_bias=False, trainable=False)(split[11])
    x13    = keras.layers.Dense(output_length, weights=[weights], use_bias=False, trainable=False)(split[12])
    x14    = keras.layers.Dense(output_length, weights=[weights], use_bias=False, trainable=False)(split[13])
    x15    = keras.layers.Dense(output_length, weights=[weights], use_bias=False, trainable=False)(split[14])
    x16    = keras.layers.Dense(output_length, weights=[weights], use_bias=False, trainable=False)(split[15])
    out    = tf.keras.layers.concatenate([x1, x2], axis=1)
    out    = tf.keras.layers.concatenate([out, x3], axis=1)
    out    = tf.keras.layers.concatenate([out, x4], axis=1)
    out    = tf.keras.layers.concatenate([out, x5], axis=1)
    out    = tf.keras.layers.concatenate([out, x6], axis=1)
    out    = tf.keras.layers.concatenate([out, x7], axis=1)
    out    = tf.keras.layers.concatenate([out, x8], axis=1)
    out    = tf.keras.layers.concatenate([out, x9], axis=1)
    out    = tf.keras.layers.concatenate([out, x10], axis=1)
    out    = tf.keras.layers.concatenate([out, x11], axis=1)
    out    = tf.keras.layers.concatenate([out, x12], axis=1)
    out    = tf.keras.layers.concatenate([out, x13], axis=1)
    out    = tf.keras.layers.concatenate([out, x14], axis=1)
    out    = tf.keras.layers.concatenate([out, x15], axis=1)
    out    = tf.keras.layers.concatenate([out, x16], axis=1)
    model  = tf.keras.models.Model(inputs=[input1], outputs=out)
    return model

  elif model_name == "vs_model":
    (input_length, output_length) = paras
    return keras.Sequential([
      keras.layers.Flatten(input_shape=(1,)),
      keras.layers.Dense(output_length, weights=[weights], use_bias=False, trainable=False),
    ])    
  elif model_name == "add_model":
    (input_length, output_length) = paras
    input1 = keras.layers.Input(shape=(input_length*2, output_length))
    split  = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=1))(input1)
    x1     = keras.layers.Add()(split)
    #input2 = keras.layers.Input(shape=(input_length,))
    #x1     = keras.layers.Add()([input1, input2])   
    model  = tf.keras.models.Model(inputs=[input1], outputs=x1)
    return model
  elif model_name == "black_model":
    (input_length, output_length) = paras
    input1 = keras.layers.Input(shape=(input_length*4, output_length))
    split  = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=1))(input1)
  # 0: r, 1: v, 2: t, 3:logV
    v2     = keras.layers.Multiply()([split[1], split[1]])
    term1  = keras.layers.Add()([v2, split[0]])
    term2  = keras.layers.Multiply()([term1, split[2]])
    term3  = keras.layers.Add()([term2, split[3]])
    model  = tf.keras.models.Model(inputs=[input1], outputs=term3);
    #model  = tf.keras.models.Model(inputs=[input1], outputs=term3);
    return model
  elif model_name == "sub_model":
    (input_length, output_length) = paras
    input1 = keras.layers.Input(shape=(input_length*2, output_length))
    split  = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=1))(input1)
    x1     = keras.layers.Subtract()(split)
    model  = tf.keras.models.Model(inputs=[input1], outputs=x1)
    return model
  elif model_name == "mul_model":
    (input_length, output_length) = paras
    input1 = keras.layers.Input(shape=(input_length*2, output_length))
    split  = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=1))(input1)
    x1     = keras.layers.multiply(split)
    model  = tf.keras.models.Model(inputs=[input1], outputs=x1)
    return model
#    input1 = keras.layers.Input(shape=(input_length,))
#    input2 = keras.layers.Input(shape=(input_length,))
#    x1     = keras.layers.multiply([input1, input1])
#    model  = tf.keras.models.Model(inputs=[input1], outputs=x1)
#    return model
  elif model_name == "log_model":
    (input_length, output_length) = paras
    input1 = keras.layers.Input(shape=(input_length,))
    x1     = keras.layers.Dense(output_length, weights=[weights], activation="sigmoid", use_bias=False, trainable=False)(input1)
#    x1     = Lambda(tf.keras.backend.log)(input1)    
    model  = tf.keras.models.Model(inputs=[input1], outputs=x1)
    return model
#  elif model_name == "max_model":
#    input1 = keras.layers.Input(shape=(input_length,))
#    split  = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=1))(input1)
#    x3     = keras.layers.Maximum()(split)
#    x3     = keras.layers.Dense(output_length, use_bias=False, trainable=False)(x3)
#    model  = tf.keras.models.Model(inputs=[input1], outputs=x3)
#    return model
  elif model_name == "tanh_model":
    (input_length, output_length) = paras
    input1 = keras.layers.Input(shape=(input_length, output_length))
    def tanh(x):
      return tf.keras.activations.tanh(x)
    x1     = Lambda(tanh)(input1)
    model  = tf.keras.models.Model(inputs=[input1], outputs=x1)
    return model
  elif model_name == "relu_model":
    (input_length, output_length) = paras
    input1 = keras.layers.Input(shape=(input_length, output_length))
    def tanh(x):
      return tf.keras.activations.relu(x)
    x1     = Lambda(tanh)(input1)
    model  = tf.keras.models.Model(inputs=[input1], outputs=x1)
    return model
  elif model_name == "max_model":
    (input_length, output_length) = paras
    input1 = keras.layers.Input(shape=(input_length, output_length, 1))
    x1     = keras.layers.MaxPooling2D([input_length, output_length/2])(input1)
    model  = tf.keras.models.Model(inputs=[input1], outputs=x1)
    return model
  elif model_name == "maxpool_model":
    (input_width, input_height, pool_width, pool_height) = paras
    input1 = keras.layers.Input(shape=(input_width, input_height, 1))
    x1     = keras.layers.MaxPooling2D(pool_size=(pool_width, pool_height), strides=(pool_width, pool_height))(input1)
    model  = tf.keras.models.Model(inputs=[input1], outputs=x1)
    return model
  elif model_name == "mean_model":
    (input_length, output_length) = paras
    input1 = keras.layers.Input(shape=(input_length, output_length, 1))
    x1     = keras.layers.AveragePooling2D([input_length, output_length/2])(input1)
#    x1     = Lambda(tf.keras.backend.mean(input1, axis=0))
    model  = tf.keras.models.Model(inputs=[input1], outputs=x1)
    return model
#    (input_length, output_length) = paras
#    input1 = keras.layers.Input(shape=(input_length*2, output_length))
#    split  = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=1))(input1)
#    x1     = keras.layers.Subtract()(split)
#   model  = tf.keras.models.Model(inputs=[input1], outputs=x1)
#i    return model
  elif model_name == "transpose_model": #operation not supported by edgetpu
    (input_length, output_length) = paras
    input1 = keras.layers.Input(shape=(input_length, output_length))
    x1     = Lambda(tf.keras.backend.transpose)(input1)
    model  = tf.keras.models.Model(inputs=[input1], outputs=x1)
    return model
  elif model_name == "crop_model":
    (input_length, output_length, block_row_len, block_col_len, start_i, start_j) = paras    
    def striding(inputs):
      return tf.strided_slice(inputs,[0, start_i,start_j],[1, start_i+block_row_len,start_j+block_col_len],[1,1,1])
    input1 = keras.layers.Input(shape=(input_length, output_length))
    x      = Lambda(striding)(input1)
    model  = tf.keras.models.Model(inputs=[input1], outputs=x)
    print(input_length, output_length, block_row_len, block_col_len, start_i, start_j)
    return model
  elif model_name == "ext_model":
    (input_length, output_length, block_row_len, block_col_len, start_i, start_j) = paras
    input1 = keras.layers.Input(shape=(block_row_len, block_col_len))
    up    = start_i
    down  = input_length  - start_i - block_row_len
    right = output_length - start_j - block_col_len
    left  = start_j
    def ext(inputs):
      return tf.pad(inputs[0], [[up, down], [left, right]])
    x      = Lambda(ext)(input1)
    model   = tf.keras.models.Model(inputs=[input1], outputs=x)
    return model
  elif model_name == "conv_model":
    (IN_W, IN_H, IN_C, OUT_C, F_W, F_H, S_W, S_H, padding, fold, multi_model) = paras
    if padding == 'none': # valid padding
# ===== test of deep conv stacking =====
      #w = np.random.randint(1, 2, size=(F_W,F_H,IN_C,OUT_C)) # unique weight for each j,k 
      #input0 = keras.layers.Input(shape=(IN_W, IN_H, IN_C), name='in0')
      #conv0 = Conv2D(filters=OUT_C, kernel_size=(F_W, F_H), strides=(S_W, S_H), activation='linear', weights=[w], use_bias=False, trainable=False)(input0)
      #conv1 = Reshape((64, 64, 4))(conv0)  
      #w2 = np.random.randint(1, 2, size=(1, 1, 256, 4)) # unique weight for each j,k 
      #conv2 = Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1), activation='linear', weights=[w2], use_bias=False, trainable=False)(conv0)

      #model = Model(inputs=[input0], outputs=conv0)
      #return model

# ===== internal tiling algorithm along with 256x64x256 shape mm2conv kernel (2b input per block) =====
# mm2conv: the conv shape:
#  input tensor shape: 64x64x4 ->( 4 2bit length 64x64 blocks)
# weight tensor shape: 8x2x4x256
# output tensor shape: 8x32x256 -> (256x256 in 2D matrix)
      #J = 8
      #K = 8

      #input0 = keras.layers.Input(shape=(IN_W, IN_H, IN_C*J), name='in0')
      #activation  = Lambda(lambda x: tf.split(x, num_or_size_splits=J, axis=3))(input0)

      #conv_list = []
      #for k in range(K):
      #  for j in range(J):
      #    w = np.random.randint(1, 2, size=(F_W,F_H,IN_C,OUT_C)) # unique weight for each j,k 
      #    conv0 = Conv2D(filters=OUT_C, kernel_size=(F_W, F_H), strides=(S_W, S_H), activation='linear', weights=[w], use_bias=False, trainable=False)(activation[j])
      #    conv_list.append(conv0)  
      #    print("j: ",j,",k: ",k)
      #out = concatenate(conv_list, axis=1)
      #model = Model(inputs=[input0], outputs=out)
      #return model
# ===== simple ===========================================================================
      if multi_model == 0:
        input0 = keras.layers.Input(shape=(IN_W, IN_H, IN_C), name='in0')
        conv0 = Conv2D(filters=OUT_C, kernel_size=(F_W, F_H), strides=(S_W, S_H), activation='linear', weights=[weights], use_bias=False, trainable=False)(input0)
      #pool_width = pool_height = 2
      #conv0 = keras.layers.AveragePooling2D(pool_size=(pool_width, pool_height), strides=(pool_width, pool_height))(conv0)
      #conv0 = keras.layers.MaxPooling2D(pool_size=(pool_width, pool_height), strides=(pool_width, pool_height))(conv0)

        model = Model(inputs=[input0], outputs=conv0)
        return model
# ===== multiple conv share the same input tensor (a consecutive blocks of tiling algo.) =====
      #if multi_model == 1:
      #  input0 = keras.layers.Input(shape=(IN_W, IN_H, IN_C), name='in0')
      #  if fold == 1:
      #    out= Conv2D(filters=OUT_C, kernel_size=(F_W, F_H), strides=(S_W, S_H), activation='linear', weights=[weights[0]], use_bias=False, trainable=False)(input0)
      #  else:
      #    conv_list = []
      #    for i in range(fold):
      #      conv0 = Conv2D(filters=OUT_C, kernel_size=(F_W, F_H), strides=(S_W, S_H), activation='linear', weights=[weights[i]], use_bias=False, trainable=False)(input0)
      #      conv_list.append(conv0)    
      #    out = concatenate(conv_list, axis=3)
      #  model = Model(inputs=[input0], outputs=out)
      #  return model
# ===== multiple conv share the same input tensor (split out OUT_C)  =====
      #input0 = keras.layers.Input(shape=(IN_W, IN_H, IN_C), name='in0')
      #w = np.random.randint(1, 2, size=(F_W,F_H,IN_C,OUT_C//fold)) 
      #if fold == 1:
      #  out= Conv2D(filters=OUT_C, kernel_size=(F_W, F_H), strides=(S_W, S_H), activation='linear', weights=[weights], use_bias=False, trainable=False)(input0)
      #else:
      #  conv_list = []
      #  for i in range(fold):
      #    conv0 = Conv2D(filters=OUT_C//fold, kernel_size=(F_W, F_H), strides=(S_W, S_H), activation='linear', weights=[w], use_bias=False, trainable=False)(input0)
      #    conv_list.append(conv0)    
      #  out = concatenate(conv_list, axis=3)
      #model = Model(inputs=[input0], outputs=out)
      #return model
# ===============================
# 2 separate conv2 taking the same input shape and concate is wat slower than just run 2 times (x 100 up)
# ===== seems good but error =============================================================
      #input_list = []
      #conv_list  = []
      #for i in range(fold):
      #  input0 = keras.layers.Input(shape=(IN_W, IN_H, IN_C))
      #  input_list.append(input0)
      #  conv = Conv2D(filters=OUT_C, kernel_size=(F_W, F_H), strides=(S_W, S_H), activation='linear', weights=[weights], use_bias=False, trainable=False)(input0)
      #  conv_list.append(conv)
      #out    = concatenate(conv_list, axis=1)
      #model = tf.keras.models.Model(inputs=input_list, outputs=[out])
      #return model
# ===== split and concate version =====================================================
      input0 = keras.layers.Input(shape=(IN_W*fold, IN_H, IN_C))
      if fold == 1:
        out = Conv2D(filters=OUT_C, kernel_size=(F_W, F_H), strides=(S_W, S_H), activation='linear', weights=[weights[0]], use_bias=False, trainable=False)(input0)
      else:
        split  = Lambda(lambda x: tf.split(x, num_or_size_splits=fold, axis=1))(input0)
        conv_list = []
        for i in range(fold):
          conv0 = Conv2D(filters=OUT_C, kernel_size=(F_W, F_H), strides=(S_W, S_H), activation='linear', weights=[weights[i]], use_bias=False, trainable=False)(split[i])
          conv_list.append(conv0)
        out    = concatenate(conv_list, axis=1)
      model = tf.keras.models.Model(inputs=[input0], outputs=out)
      return model
    # for hotspot3D usage 
    elif padding == 'SAME': # same padding means input & output feature map have the same size with padding zeros
      input0 = Input(shape=(IN_W,IN_H,IN_C))
      conv = Conv2D(filters=OUT_C, kernel_size=(F_W,F_H), strides=(1,1), padding="same", activation='linear', weights=[weights[i]], use_bias=False, trainable=False)(input0)
      model = Model(inputs=[input0], outputs=[conv])
      return model
    else:
      print("undefined padding type: "+padding)
      exit(0)
  elif model_name == "depthwise_conv_model":
    (IN_W, IN_H, IN_C, OUT_C, F_W, F_H, S_W, S_H, padding) = paras
    IN_W = 512
    IN_H = 512
    IN_C = 32
    OUT_C = 32
    F_W = S_W = 16
    F_H = S_H = 16
    input0 = Input(shape=(IN_W, IN_H, IN_C))
    conv = tf.keras.layers.DepthwiseConv2D(kernel_size=(F_W, F_H), strides=(S_W, S_H), padding='valid', depth_multiplier=OUT_C, dilation_rate=(1, 1), activation='linear', use_bias=False, trainable=False)(input0)
    model = Model(inputs=[input0], outputs=[conv])
    model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
    return model
# other padding are: replication padding, reflection padding, constant padding

                                                 
def generate_tflite_v2(target, w, model_name, outfile_name, ramdisk, paras, default_range=(0, 255), train_size=1, EPOCH=1, stats_values=(0, 1)):    
  
  frozen_model_path = "./"+model_name+".pb"
  saved_model_path  = "./"+model_name+".h5"
  #frozen_model_path = "~/GPTPU/data/"+model_name+".pb"
  tflite_model_name = model_name+".tflite"
  quant_model_name = model_name+"_quant.tflite"
  output_dir = model_name+"_tflite"
  
  model = build_keras_model(w, paras, model_name)
  model.summary()
# ===== QAT qunatization =======
#  model = quantize_model(model)
#  model.summary()
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# ===== set weights =====
#  print(dir(model.layers[2]))
#  model.layers[2].set_weights([w])
# ======================

  model.save(saved_model_path)

  IN_W = 8
  IN_H = 128
  IN_C = 64
  OUT_C = 64

  def data_gen():
    img = np.random.random_sample((1, IN_W, IN_H, IN_C))
    img = np.array(img, dtype=np.float32)
    yield [img]
# ===== post-training qunatization =====
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
#  converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(saved_model_path)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
# ===== the 16x8 flag =====
  print(tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8)
  converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
#  converter.post_training_qunatize = True
#  converter.representative_dataset = data_gen
#  converter.inference_type = tf.compat.v1.lite.constants.QUANTIZED_UINT8
#  converter.default_ranges_stats = (0, 255) # output representation range, we must need 0, 255
#  input_arrays = converter.get_input_arrays()
#  converter.quantized_input_stats = {input_arrays[0]: (0., 1.)} # mean, stddev
# ===== input real value = (quantized input - mean) / std dev

  converter.experimental_new_converter = False

  tflite_model = converter.convert()
  open(quant_model_name, 'wb').write(tflite_model)
  print("converter.convert() done.")

  out_path = "~/GPTPU/data/" if ramdisk == 0 else "/mnt/ramdisk/"

  # Compile the tflite to edgetpu compatible one. -s enables all converted/unconverted operations
  print("quant_model_name now is at: " + quant_model_name)
  if model_name == "conv_model":
    print("edgetpu_compiler is compiling to xxx_quant_edgetpu.tflite")
 #   (IN_W, IN_H, IN_C, OUT_C, F_W, F_H, S_W, S_H) = paras
 #   params = str(IN_W)+"x"+str(IN_H)+"x"+str(IN_C)+"x"+str(F_W)+"x"+str(F_H)+"x"+str(S_W)+"x"+str(S_H)
 #   print("cp "+out_path+"conv_model_tflite/"+quant_model_name+" "+out_path+"conv_model_tflite/conv_model_quant_"+params+".tflite")
    print(    "edgetpu_compiler "+quant_model_name+" -o "+out_path+output_dir+" -s")
    os.system("edgetpu_compiler "+quant_model_name+" -o "+out_path+output_dir+" -s")
    print(    "sudo mv -f "+out_path+output_dir+"/"+model_name+"_quant_edgetpu.tflite"+" "+outfile_name)
    os.system("sudo mv -f "+out_path+output_dir+"/"+model_name+"_quant_edgetpu.tflite"+" "+outfile_name)
  elif model_name == "imv_model":
    print("edgetpu_compiler "+quant_model_name+" -o "+"./"+" -s")
    os.system("edgetpu_compiler "+quant_model_name+" -o "+"./"+" -s")
    #print("sudo mv -f "+quant_model_name+" "+out_path+output_dir)  
    #os.system("sudo mv -f "+quant_model_name+" "+out_path+output_dir)  
    #print("sudo mv -f "+out_path+output_dir+"/"+quant_model_name+" "+outfile_name)  
    #os.system("sudo mv -f "+out_path+output_dir+"/"+quant_model_name+" "+outfile_name)  
    #print("finally model is at: " + outfile_name)
  else:
    #print("edgetpu_compiler "+quant_model_name+" -o "+out_path+output_dir+" -s")
    #os.system("edgetpu_compiler "+quant_model_name+" -o "+out_path+output_dir+" -s")
    print("sudo mv -f "+quant_model_name+" "+out_path+output_dir)  
    os.system("sudo mv -f "+quant_model_name+" "+out_path+output_dir)  
    print("sudo mv -f "+out_path+output_dir+"/"+quant_model_name+" "+outfile_name)  
    os.system("sudo mv -f "+out_path+output_dir+"/"+quant_model_name+" "+outfile_name)  
    print("finally model is at: " + outfile_name)
  
  os.system("rm "+saved_model_path)

def save_weight(w, file_name):
  if os.path.isfile("../data/"+str(file_name)+".npy"): # remove old one if exist
    os.system("rm ../data/"+str(file_name)+".npy")
  np.save("../data/"+str(file_name), w)

if __name__ == "__main__":
  if tf.__version__ != '1.13.1':
    print(" required tf version: 1.13.1, you have ", tf.__version__, " instead.")
#    exit(0)

  # model name should from argument
  parser = argparse.ArgumentParser()
  parser.add_argument("--platform"    , action='store', dest='target'       , default='m2',          help='specify platform(m2, coral)')
  parser.add_argument("--model"       , action='store', dest='model_name'   , default='conv_model',  help='specify model name')
  parser.add_argument("--in_w_name"   , action='store', dest='in_w_name'    , default='./../mm2conv_weight.txt',  help='specify input weight path/name')
  parser.add_argument("--out_scale"   , action='store', dest='out_scale'    , default='1',           help='specify output value scale')
  parser.add_argument("--data_type"   , action='store', dest='data_type'     , default='uint8',       help='uint8/int8')
  parser.add_argument("--outfile_name", action='store', dest='outfile_name' , default='del.tflite',  help='specify outfile path/name')
  parser.add_argument("--in_size"     , action='store', dest='input_length' , default='1024',         help='input length of model input')
  parser.add_argument("--out_size"    , action='store', dest='output_length', default='1024',         help='output length of model output')
  parser.add_argument("--blk_row"     , action='store', dest='block_row_len', default='16',          help='length of block row size')
  parser.add_argument("--blk_col"     , action='store', dest='block_col_len', default='16',          help='length of block col size')
  parser.add_argument("--start_i"     , action='store', dest='start_i'      , default='0',           help='starting i index')
  parser.add_argument("--start_j"     , action='store', dest='start_j'      , default='0',           help='starting j index')
  parser.add_argument("--ramdisk"     , action='store', dest='ramdisk'      , default='1',           help='using ramdisk')
  parser.add_argument("--IN_W"        , action='store', dest='IN_W'         , default='8',         help='input feature map row size')
  parser.add_argument("--IN_H"        , action='store', dest='IN_H'         , default='128',         help='input feature map col size')
  parser.add_argument("--IN_C"        , action='store', dest='IN_C'         , default='64',           help='input model channel')
  parser.add_argument("--OUT_C"       , action='store', dest='OUT_C'        , default='256',        help='output feature map channel')
  parser.add_argument("--F_W"         , action='store', dest='F_W'          , default='2',         help='filter row size')
  parser.add_argument("--F_H"         , action='store', dest='F_H'          , default='2',           help='filter col size')
  parser.add_argument("--S_W"         , action='store', dest='S_W'          , default='2',         help='stride row direction')
  parser.add_argument("--S_H"         , action='store', dest='S_H'          , default='2',           help='stride col direction')
  parser.add_argument("--PADDING"     , action='store', dest='PADDING'      , default='none',        help='choose either \'none\' or \'SAME\' or \'replication\'')
  parser.add_argument("--ITER"        , action='store', dest='ITER'         , default='1',           help='iter for gptpu_imv()')
  parser.add_argument("--mm256blk"    , action='store', dest='mm256blk'     , default='1',           help='boolean mode for enabling mm256blk or not (exact mode only)')
  parser.add_argument("--fold"        , action='store', dest='fold'         , default='1',           help='# of independent and identical models in one kernel)')
  parser.add_argument("--multi_model" , action='store', dest='multi_model'  , default='0',           help='enable the mode for multi model sharing the same input tensor')

  args         = parser.parse_args()
  target       = args.target
  model_name   = args.model_name
  outfile_name = args.outfile_name 
  in_w_name    = args.in_w_name
  out_scale    = args.out_scale
  data_type    = args.data_type
  ramdisk      = int(args.ramdisk)
  ITER         = int(args.ITER)
  mm256blk     = int(args.mm256blk)
  fold         = int(args.fold)
  multi_model  = int(args.multi_model)
  padding      = args.PADDING
  if model_name == "mv_model" or model_name == "mm_model":
    input_length  = int(args.input_length)
    output_length = int(args.output_length)
    paras = (input_length, output_length)
  elif model_name == "push_model":  # particle push model
    input_length = int(args.input_length)
    paras = (input_length)
  elif model_name == "imv_model":
    input_length  = int(args.input_length)
    output_length = int(args.output_length)
    paras = (input_length, output_length, ITER)
  elif model_name == "bmv_model":
    input_length  = int(args.input_length)
    output_length = int(args.output_length)
    paras = (input_length, output_length)
  elif model_name == "vs_model":
    input_length  = 1
    output_length = int(args.output_length)
    paras = (input_length, output_length)
  elif model_name == "add_model":
    input_length  = int(args.input_length)
    output_length = int(args.output_length)
    paras = (input_length, output_length)
  elif model_name == "sub_model":
    input_length  = int(args.input_length)
    output_length = int(args.output_length)
    paras = (input_length, output_length)
  elif model_name == "mul_model":
    input_length  = int(args.input_length)
    output_length = int(args.output_length)
    paras = (input_length, output_length)
  elif model_name == "log_model":
    input_length = int(args.input_length)
    output_length = int(args.output_length)
    paras = (input_length, output_length)
  elif model_name == "tanh_model":
    input_length = int(args.input_length)
    output_length = int(args.output_length)
    paras = (input_length, output_length)
  elif model_name == "relu_model":
    input_length = int(args.input_length)
    output_length = int(args.output_length)
    paras = (input_length, output_length)
  elif model_name == "maxpool_model":
    input_width   = int(args.IN_W)
    input_height  = int(args.IN_H)
    pool_width    = int(args.F_W)
    pool_height   = int(args.F_H)
    paras = (input_width, input_height, pool_width, pool_height)
  elif model_name == "mean_model":
    input_length  = int(args.input_length)
    output_length = int(args.output_length)
    paras = (input_length, output_length)  
  elif model_name == "max_model":
    input_length  = int(args.input_length)
    output_length = int(args.output_length)
    paras = (input_length, output_length)  
  elif model_name == "min_model":
    input_length  = int(args.input_length)
    output_length = int(args.output_length)
    paras = (input_length, output_length)  
  elif model_name == "transpose_model":
    input_length  = int(args.input_length)
    output_length = int(args.output_length)
    paras = (input_length, output_length)  
  elif model_name == "crop_model":
    input_length  = int(args.input_length)
    output_length = int(args.output_length)
    block_row_len = int(args.block_row_len)
    block_col_len = int(args.block_col_len)
    start_i       = int(args.start_i)
    start_j       = int(args.start_j)
    paras = (input_length, output_length, block_row_len, block_col_len, start_i, start_j)  
  elif model_name == "ext_model":
    input_length  = int(args.input_length)
    output_length = int(args.output_length)
    block_row_len = int(args.block_row_len)
    block_col_len = int(args.block_col_len)
    start_i       = int(args.start_i)
    start_j       = int(args.start_j)
    paras = (input_length, output_length, block_row_len, block_col_len, start_i, start_j)  
  elif model_name == "conv_model" or model_name == "depthwise_conv_model":
    IN_W  = int(args.IN_W)
    IN_H  = int(args.IN_H)
    IN_C  = int(args.IN_C)
    OUT_C = int(args.OUT_C)
    F_W   = int(args.F_W)
    F_H   = int(args.F_H)
    S_W   = int(args.S_W)
    S_H   = int(args.S_H)
    paras = (IN_W, IN_H, IN_C , OUT_C, F_W, F_H, S_W, S_H, padding, fold, multi_model)  
  elif model_name == "black_model":
    input_length  = int(args.input_length)
    output_length = int(args.output_length)
    paras = (input_length, output_length)  
  else:
    pass

  weight_name = model_name+"_weight"
  # ===== start main program =====
  print(model_name)
  np.random.seed(2346)
  if model_name == "mv_model" or model_name == "mm_model":
    w = np.random.randint(-1, 2, size=(input_length, output_length)) / 1
    w = np.random.random_sample((input_length, output_length))
    print(w)
    save_weight(w, str(weight_name))
  elif model_name == "push_model":
    w = 1
  elif model_name == "bmv_model":
    w = np.random.randint(3, 4, size=(input_length, output_length)) / 1
  elif model_name == "vs_model":
    w = np.random.randint(1, 2, size=(1, output_length)) / 1
    save_weight(w, str(weight_name))
  elif model_name == "add_model":
    w = 1
  elif model_name == "sub_model":
    w = 1
  elif model_name == "mul_model":
    w = 1
  elif model_name == "log_model":
    w = 1    
    w = np.random.random_sample((input_length, output_length)) / 1
    save_weight(w, str(weight_name))
  elif model_name == "tanh_model":
    w = 1
  elif model_name == "relu_model":
    w = 1
  elif model_name == "maxpool_model":
    w = 1
  elif model_name == "max_model":
    w = 1
  elif model_name == "min_model":
    w = 1
  elif model_name == "mean_model":
    w = 1
  elif model_name == "transpose_model":
    w = 1
  elif model_name == "crop_model":
    w = 1
  elif model_name == "ext_model":
    w = 1
  elif model_name == "conv_model":
    print("multi_model: "+ str(multi_model == 0))
    if multi_model == 0:
      if mm256blk == 0:
        w = np.random.randint(1, 2, size=(F_W,F_H,IN_C,OUT_C)) 
        with open(in_w_name, "rb") as f:
          a = np.fromfile(f, dtype=np.int32)
          print(a.shape)
          print("OUT_C: ", OUT_C, ", F_W: ", F_W, ", F_H: ", F_H, ", IN_C: ", IN_C)
          a = np.reshape(a, (OUT_C, F_W, F_H, IN_C)) 
          a = np.transpose(a, (1, 2, 3 ,0))
          w = a
          print("1: ", (w == 1).sum(), "0: ", (w== 0).sum())
      else: # 256mm block design
        w = np.random.randint(1, 2, size=(F_W,F_H,IN_C,OUT_C)) 
        #with open(in_w_name, "rb") as f:
        #  a = np.fromfile(f, dtype=np.int32)
        #  print(a.shape)
        #  print("OUT_C: ", OUT_C, ", F_W: ", F_W, ", F_H: ", F_H, ", IN_C: ", IN_C)
        #  a = np.reshape(a, (OUT_C, F_W, F_H, IN_C)) 
        #  a = np.transpose(a, (1, 2, 3 ,0))
        #  w = a
        #  print(w.shape, w)
    else: # multi_model is on
      print("skip")
      w = []
      #names = glob.glob(in_w_name+"/*")
      #print(in_w_name, names, len(names))
      for i in range(fold):
      #  with open(names[i], "rb") as f: # now in_w_name now serve as a directory contains multiple model parameter sets
      #    a = np.fromfile(f, dtype=np.int32)
      #    print(a.shape)
      #    print("OUT_C: ", OUT_C, ", F_W: ", F_W, ", F_H: ", F_H, ", IN_C: ", IN_C)
      #    a = np.reshape(a, (OUT_C, F_W, F_H, IN_C)) 
      #    a = np.transpose(a, (1, 2, 3 ,0))
      #    w.append(a)
        w.append(np.random.randint(1, 2, size=(F_W, F_H, IN_C, OUT_C)))
  elif model_name == "imv_model":
    #w = np.random.random(0, 2, size=(input_length, output_length)) 
    with open(in_w_name, "rb") as f:
      w = np.fromfile(f, dtype=np.double)
      w = np.reshape(w, (1, 1, input_length, output_length)) 

  else:
    w = 1

  if data_type == 'int8': 
    stat_values = (128, out_scale)
    the_range   = (-128, 127)
  elif data_type == 'uint8':
    stat_values = (0, out_scale)
    the_range   = (0, 255)
  else:
    print("invalid data type: " + data_type)
    exit(1)

  generate_tflite_v2(target, w, model_name, outfile_name, ramdisk, paras, default_range = the_range, stats_values = stat_values)

    

#
