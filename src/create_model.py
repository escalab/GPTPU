import os
import glob
import time
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('INFO')
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_util 
import warnings
import argparse
import random

# surpass tf FutureWarning
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

print(tf.version)

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
    
    input1 = keras.layers.Input(shape=(1024, 1024, 3), name='in0')
    
    split  = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input1)
    delta = split[0]        
    ly    = split[1]        
    oldw  = split[2]        

   
    a     = keras.layers.multiply([delta, ly]) # (2) -> (4)
    b     = keras.layers.Add()([a, oldw])  # (1)

    model  = tf.keras.models.Model(inputs=[input1], outputs=b)
    return model
  
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
 #   (IN_W, IN_H, IN_C, OUT_C, F_W, F_H, S_W, S_H, padding, fold, multi_model) = paras
#    IN_W = 8
#    IN_H = 128
#    IN_C = 64
#    OUT_C = 256
#    F_W = S_W = 2 
#    F_H = S_H = 2
#    weights = np.random.randint(1, 256, size=(F_W,F_H,IN_C,OUT_C)) 
#    input0 = keras.layers.Input(shape=(IN_W, IN_H, IN_C), name='in0')
#    conv0 = Conv2D(filters=OUT_C, kernel_size=(F_W, F_H), strides=(S_W, S_H), activation='linear', weights=[weights], use_bias=False, trainable=False)(input0)
      #pool_width = pool_height = 2
      #conv0 = keras.layers.AveragePooling2D(pool_size=(pool_width, pool_height), strides=(pool_width, pool_height))(conv0)
      #conv0 = keras.layers.MaxPooling2D(pool_size=(pool_width, pool_height), strides=(pool_width, pool_height))(conv0)

#    model = Model(inputs=[input0], outputs=conv0)
#    return model

    (input_length, output_length, ITER) = paras
    if input_length != output_length:
      print("create model: imv only support square weight matrix for now, in: "+str(input_length)+", out: " +str(output_length))
      exit(0)
    print("in size: ", input_length, ", out size: ", output_length)

    IN_W = 1#8
    IN_H = 1#128
    IN_C = input_length #64
    OUT_C = output_length #256
    F_W = S_W = 1#2
    F_H = S_H = 1#2

    np.random.seed(9487)
    if 0: #os.path.exists("./pagerank_1K_iter5_weight.txt"):
      pass
   #   w = np.fromfile("./pagerank_1K_iter5_weight.txt", dtype=np.float32)
   #   print("weight matrix from file");
#    w = np.load("../data/pagerank_1K_iter5_weight.npy")
#    w = np.expand_dims(w, axis=0)
#    w = np.expand_dims(w, axis=0)
#    print(w, w.shape)
    else:
      print("weight matrix from new generating...");
      w = np.random.randint(0, 2, size=(F_W, F_H, IN_C, OUT_C))
# ===== use pagerank specific weight ======
      w = w.astype(np.float32)
      min_n = OUT_C
      for i in range(OUT_C):
        n = (w[:,:,:,i] == 1).sum()
        min_n = min(min_n, n)
        w[:,:,:,i] = w[:,:,:,i] / n
# ===== write weight to file in bin =====#
      w.tofile("./pagerank_1K_iter1_weight.txt", format="%s")
#    print(w)
# ===== WARN test ====
    w = np.swapaxes(w, 2, 3)
    w = w * 255.0 * min_n
    w= w.astype(np.uint8)
#    for i in range(1):
#      w[0, 0, i, i] = 1
    #weights = np.reshape(weights, (F_W, F_H, IN_C, OUT_C))
    print(w, w.shape)
    input0 = keras.layers.Input(shape=(IN_W, IN_H, IN_C), name='in0')
    layer = input0

    def myconv(x):
      layer = tf.nn.conv2d(input=x, filter=w, strides=[1, 1, S_W, S_H], padding='VALID')
      return layer
    for i in range(ITER):
      #layer = Conv2D(filters=OUT_C, kernel_size=(F_W, F_H), strides=(S_W, S_H), activation='linear', weights=[w], use_bias=False, trainable=False)(layer)
      layer = Lambda(myconv)(layer) # avoid uninitialzed byg

    model = Model(inputs=[input0], outputs=layer)
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
    (IN_W, IN_H, IN_C, OUT_C, F_W, F_H, S_W, S_H, padding, fold, multi_model, ITER) = paras
    print("padding: ", padding, ", multi_model: ", multi_model)
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
        print("IN_W: ", IN_W, ", IN_H: ", IN_H, ", IN_C: ", IN_C, ", OUT_C: ", OUT_C, ", F_W: ", F_W, ", F_H: ", F_H)
        print("weight.shape: ", weights.shape)
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
#      input0 = Input(shape=(IN_W,IN_H,IN_C))
#      conv = Conv2D(filters=OUT_C, kernel_size=(F_W,F_H), strides=(1,1), padding="same", activation='linear', weights=[weights[i]], use_bias=False, trainable=False)(input0)
#      model = Model(inputs=[input0], outputs=[conv])
#      return model
# ===== multi-layer conv model ===== (for hotspot benchmark)
    
      input0 = keras.layers.Input(shape=(IN_W, IN_H, IN_C*2), name='in0')
      print(input0)
      print("input0 shape: ", input0.shape)
      split  = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(input0)
      add_w  = split[1]
      layer  = split[0]
      #print("input layer shape: ", layer.shape)

#      layer = input0

      wc = np.random.randint(0, 1, size=(F_W, F_H, 1, 1))
      wc = wc.astype(np.uint8)
      #wc[0,1,0,0] = 76
      #wc[1,0,0,0] = 76
      #wc[1,2,0,0] = 76
      #wc[2,1,0,0] = 76
      #wc[1,1,0,0] = 255
        
      print("input shape: ", input0.shape)
      print(wc)
      print("wc shape: ", wc.shape)
      #print("add_w shape: ", add_w.shape)
      #print("IN_W: ", IN_W, "IN_H: ", IN_H, "F_W: ", F_W, "F_H: ", F_H, "IN_C: ", IN_C, "OUT_C: ", OUT_C, "S_W: ", S_W, "S_H: ", S_H)
      def myconv_s(xx):
        layer = tf.nn.conv2d(input=xx, filter=wc, strides=[1, 1, 1, 1], padding='SAME')
        #print("xx.shape: ", xx.shape, ", wc: ", wc.shape)
        #print("layer.shape: ", layer.shape)
        #layer = tf.math.add(layer, add_w)
        #layer = tf.nn.depthwise_conv2d(input=x, filter=wc, strides=[1,1,1,1], padding='SAME', name=None)
        #layer = tf.keras.layers.DepthwiseConv2D(kernel_size=(F_W, F_H), strides=(1, 1), padding='same', depth_multiplier=OUT_C, activation='linear', use_bias=False, trainable=False)(x)
        return layer
      def myadd(x):
        layer = tf.math.add(x, add_w)
        return layer
 
      #blk_cnt = 16
      #cols_list = []
      #blk_list  = []

      for i in range(ITER):
        layer = Lambda(myconv_s)(layer) # avoid uninitialzed bug
# split + concate for trying multiple filter design
#        cols = Lambda(lambda x: tf.split(x, num_or_size_splits=blk_cnt, axis=-2))(layer)
#        for j in range(blk_cnt):
#          blk_list.append([])
#          blks = Lambda(lambda x: tf.split(x, num_or_size_splits=blk_cnt, axis=-3))(cols[j])
#          for k in range(blk_cnt):
#            blk = Lambda(myconv_s)(blks[k])
#            blk_list[j].append(blk) 
#          col = keras.layers.concatenate(blk_list[j], axis=-3)
#          cols_list.append(col)        
#        layer = keras.layers.concatenate(cols_list, axis=-2)

        layer = Lambda(myadd)(layer)

      model = Model(inputs=[input0], outputs=layer)
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
                                                 
def press_to_zero(outfile_name, start, end):
  f = open(outfile_name, "r+b")
  for  i in range(start, end):
    f.seek(i)
    byte = f.read(1)
    byte = hex(int.from_bytes(byte, byteorder='big'))
    if byte == hex(0x01):
      f.seek(i)
      f.write(b'\x00')
  f.close()

def generate_tflite_v2(ITER, generate_method, target, w, model_name, outfile_name, ramdisk, paras, default_range=(0, 255), train_size=1, EPOCH=1, stats_values=(0, 1)):    
  #if model_name == "mv_model" or model_name == "mul_model" or model_name == "add_model":
  #elif model_name == "vs_model":
  
  frozen_model_path = "./"+model_name+".pb"
  #frozen_model_path = "~/GPTPU/data/"+model_name+".pb"
  tflite_model_name = model_name+".tflite"
  quant_model_name = model_name+"_quant.tflite"
  output_dir = model_name+"_tflite"
  check_point_path = '../data/checkpoint/checkpoints_'+model_name

  method = generate_method

  if method == 1:
    #load data ( dummy action )
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    train_size = 1
    train_images = np.random.randint(255, size=(train_size,1, 1, input_length))
    train_labels = np.asarray([random.randint(0, output_length-1) for x in train_labels]) # at least one label has to be max.
   
    train_graph = tf.Graph()
    train_sess = tf.compat.v1.Session(graph=train_graph)
    keras.backend.set_session(train_sess)
    with train_graph.as_default():
      train_model = build_keras_model(w, paras, model_name)
      train_model.summary()
      in_layer_name  = train_model.layers[0].input.name.split(':')[0] # input layer name
      out_layer_name = train_model.layers[-1].output.name.split(':')[0] # output layer name
      #tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=100)
      tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=1)
      
      train_sess.run(tf.global_variables_initializer())    
      #train_sess.run(tf.variables_initializer(tf.train.AdamOptimizer().variables()))
      #weights = np.random.randint(1, 3, size=(1, 1, 1024, 1024))
      for i in range(ITER):
        train_model.layers[i+1].set_weights([w])
      #train_model.layers[2].set_weights([weights])
      print(dir(train_model.layers[1]))
      print(train_model.layers[1].bias_initializer)
      #print(train_model.get_weights())
      train_model.compile(
          optimizer='adam',
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy']
      )
      train_model.fit(train_images[:train_size], train_labels[:train_size], epochs=EPOCH)
      # save graph and checkpoints
      saver = tf.train.Saver()
      saver.save(train_sess, check_point_path)
   
   # ====== eval  phase ==========
    eval_graph = tf.Graph()
    eval_sess = tf.compat.v1.Session(graph=eval_graph)

    keras.backend.set_session(eval_sess)

    with eval_graph.as_default():
      keras.backend.set_learning_phase(0)
      eval_model = build_keras_model(w, paras, model_name)
      tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
      eval_graph_def = eval_graph.as_graph_def()
      saver = tf.train.Saver()
      saver.restore(eval_sess, check_point_path)

      frozen_graph_def = tf.graph_util.convert_variables_to_constants(
          eval_sess,
          eval_graph_def,
          [eval_model.output.op.name]
      )
  if method == 2:
    #weights = np.random.randint(1, 2, size=(F_W, F_H, IN_C))
    # the new ========== eval phase ==========
    eval_graph = tf.Graph()
    eval_sess = tf.compat.v1.Session(graph=eval_graph)

    keras.backend.set_session(eval_sess)
    with eval_graph.as_default():
      keras.backend.set_learning_phase(0)
      eval_model = build_keras_model(w, paras, model_name)
      eval_model.summary()

#      with tf.device('/CPU:0'):
#        inn = np.zeros((1024, 1024, 3))
#        t = time.time()
#        for i in range(1000):
#          pred = eval_model(inn)
#        print("pred time: {:.9f}s".format((time.time() - t)/1000))

      in_layer_name  = eval_model.layers[0].input.name.split(':')[0] # input layer name
      out_layer_name = eval_model.layers[-1].output.name.split(':')[0] # output layer name
      tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
# ===== set weights =====
      #eval_sess.run(tf.global_variables_initializer())    
#      for i in range(ITER):
#        eval_model.layers[i+1].set_weights([w])
# =======================
# ===== print all variables =====
      #print(tf.get_default_graph().as_graph_def())
      #print(dir(tf.get_default_graph().as_graph_def()))
#      for item in [n for n in tf.get_default_graph().as_graph_def().node]:
#        print(item.name)
#        if item.name == "conv2d/act_quant/min" or item.name == "conv2d_1/Conv2D":
#          print(dir(item), type(item))
# ==============================
      eval_graph_def = eval_graph.as_graph_def()

      frozen_graph_def = tf.graph_util.convert_variables_to_constants(
          eval_sess,
          eval_graph_def,
          [eval_model.output.op.name]
        #[item.op.name for item in eval_model.output]
      )

  with open(frozen_model_path, 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())
  # =================================

  # make sure old file is not used before converting
  os.system("rm -f "+quant_model_name)

  # tflite model converter
  if target == 'coral': # coral dev board
    converter = 'tflite_convert'
  elif target == 'm2': # m2 form factor edgetpu accelerator
    converter = 'toco'
  else:
    print('target platform wrong: '+ target)
    exit()  
  print(converter+" is compiling to xxx_quant.tflite")
  os.system(converter+" --output_file="+quant_model_name+\
		" --graph_def_file="+frozen_model_path+\
		" --input_array="+in_layer_name+\
		" --output_array="+out_layer_name+\
		" --inference_type=QUANTIZED_UINT8"+\
  # default range is for output range
		" --default_ranges_min="+str(default_range[0])+\
		" --default_ranges_max="+str(default_range[1])+\
  #real_input_value = (quantized_input_value - mean_value) / std_dev_value
		" --mean_value="+str(stats_values[0])+\
		" --std_dev_values="+str(1.0/(float)(stats_values[1]))) # more intuitive
  print("std_dev_values=the scale set here is ="+str(1.0/(float)(stats_values[1])))
  #out_path = "./../data/" if ramdisk == 0 else "/mnt/ramdisk/"
  out_path = "~/GPTPU/data/" if ramdisk == 0 else "/mnt/ramdisk/"
  # Prepare output directory (not for blocking algorithm any more)
  if os.path.isdir(out_path+output_dir): # remove old one if exist
    os.system("rm -rf "+out_path+output_dir)
  os.system("mkdir "+out_path+output_dir)


  # Compile the tflite to edgetpu compatible one. -s enables all converted/unconverted operations
  print("quant_model_name now is at: " + quant_model_name)
  print("edgetpu_compiler is compiling to xxx_quant_edgetpu.tflite")
  if model_name == "conv_model":
 #   (IN_W, IN_H, IN_C, OUT_C, F_W, F_H, S_W, S_H) = paras
 #   params = str(IN_W)+"x"+str(IN_H)+"x"+str(IN_C)+"x"+str(F_W)+"x"+str(F_H)+"x"+str(S_W)+"x"+str(S_H)
 #   print("cp "+out_path+"conv_model_tflite/"+quant_model_name+" "+out_path+"conv_model_tflite/conv_model_quant_"+params+".tflite")
    print(    "edgetpu_compiler "+quant_model_name+" -o "+out_path+output_dir+" -s")
    os.system("edgetpu_compiler "+quant_model_name+" -o "+out_path+output_dir+" -s")
    print(    "sudo mv -f "+out_path+output_dir+"/"+model_name+"_quant_edgetpu.tflite"+" "+outfile_name)
    os.system("sudo mv -f "+out_path+output_dir+"/"+model_name+"_quant_edgetpu.tflite"+" "+outfile_name)
  elif model_name == "imv_model":
    print(    "edgetpu_compiler "+quant_model_name+" -o "+out_path+output_dir+" -s")
    os.system("edgetpu_compiler "+quant_model_name+" -o "+out_path+output_dir+" -s")
    print(    "sudo mv -f "+out_path+output_dir+"/"+model_name+"_quant_edgetpu.tflite"+" "+outfile_name)
    os.system("sudo mv -f "+out_path+output_dir+"/"+model_name+"_quant_edgetpu.tflite"+" "+outfile_name)

    print("finally model is at: " + outfile_name)
#    print("pressing to zero....")
#    if ITER == 5:
#      press_to_zero(outfile_name, 12812, 5275916) # 1K_iter5
#    if ITER == 4:
#      press_to_zero(outfile_name, 12812, 4223244) # 1K_iter2 0x20510c
#    if ITER == 3:
#      press_to_zero(outfile_name, 12812, 3170572) # 1K_iter2 0x20510c
#    if ITER == 2:
#      press_to_zero(outfile_name, 12812, 2117900) # 1K_iter2 0x20510c
#    if ITER == 1:
#      press_to_zero(outfile_name, 12812, 1065228) # 1K_iter1 0x10410c
#    print("finally press2zero model is at: " + outfile_name)
  else:
    #print("edgetpu_compiler "+quant_model_name+" -o "+out_path+output_dir+" -s")
    #os.system("edgetpu_compiler "+quant_model_name+" -o "+out_path+output_dir+" -s")
    print("sudo mv -f "+quant_model_name+" "+out_path+output_dir)  
    os.system("sudo mv -f "+quant_model_name+" "+out_path+output_dir)  
    print("sudo mv -f "+out_path+output_dir+"/"+quant_model_name+" "+outfile_name)  
    os.system("sudo mv -f "+out_path+output_dir+"/"+quant_model_name+" "+outfile_name)  
    print("finally model is at: " + outfile_name)
  #clean up intermediate .pb files
  os.system("rm "+frozen_model_path)

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
  parser.add_argument("--in_w_name"   , action='store', dest='in_w_name'    , default='default',  help='specify input weight path/name')
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
  parser.add_argument("--IN_W"        , action='store', dest='IN_W'         , default='1024',         help='input feature map row size')
  parser.add_argument("--IN_H"        , action='store', dest='IN_H'         , default='1024',         help='input feature map col size')
  parser.add_argument("--IN_C"        , action='store', dest='IN_C'         , default='1',           help='input model channel')
  parser.add_argument("--OUT_C"       , action='store', dest='OUT_C'        , default='1',        help='output feature map channel')
  parser.add_argument("--F_W"         , action='store', dest='F_W'          , default='3',         help='filter row size')
  parser.add_argument("--F_H"         , action='store', dest='F_H'          , default='3',           help='filter col size')
  parser.add_argument("--S_W"         , action='store', dest='S_W'          , default='1',         help='stride row direction')
  parser.add_argument("--S_H"         , action='store', dest='S_H'          , default='1',           help='stride col direction')
  parser.add_argument("--PADDING"     , action='store', dest='PADDING'      , default='none',        help='choose either \'none\' or \'SAME\' or \'replication\'')
  parser.add_argument("--ITER"        , action='store', dest='ITER'         , default='1',           help='iter for gptpu_imv()')
  parser.add_argument("--mm256blk"    , action='store', dest='mm256blk'     , default='1',           help='boolean mode for enabling mm256blk or not (exact mode only)')
  parser.add_argument("--fold"        , action='store', dest='fold'         , default='1',           help='# of independent and identical models in one kernel)')
  parser.add_argument("--multi_model" , action='store', dest='multi_model'  , default='0',           help='enable the mode for multi model sharing the same input tensor')
  parser.add_argument("--method"      , action='store', dest='generate_method', default='2',         help='generate method (1 or 2)')

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
  generate_method = int(args.generate_method)
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
    paras = (IN_W, IN_H, IN_C , OUT_C, F_W, F_H, S_W, S_H, padding, fold, multi_model, ITER)  
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
    print("multi_model == 0 ?: "+ str(multi_model == 0))
    if multi_model == 0:
      if mm256blk == 0:
        w = np.random.randint(1, 2, size=(F_W,F_H,IN_C,OUT_C)) 
#        with open(in_w_name, "rb") as f:
#          a = np.fromfile(f, dtype=np.int32)
#          print(a.shape)
#          print("OUT_C: ", OUT_C, ", F_W: ", F_W, ", F_H: ", F_H, ", IN_C: ", IN_C)
#          a = np.reshape(a, (OUT_C, F_W, F_H, IN_C)) 
#          a = np.transpose(a, (1, 2, 3 ,0))
#          w = a
#          print("w: ")
#          print(w)
#          print("1: ", (w == 1).sum(), "0: ", (w== 0).sum())
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
    IN_W = 1#8
    IN_H = 1#128
    IN_C = input_length #64
    OUT_C = output_length #256
    F_W = S_W = 1#2
    F_H = S_H = 1#2
    w = np.random.randint(0, 1, size=(F_W, F_H, IN_C, OUT_C))
    if in_w_name != "default":
      with open(in_w_name, "rb") as f:
        a = np.fromfile(f, dtype=np.uint8)
        print(a.shape)
        print("OUT_C: ", OUT_C, ", F_W: ", F_W, ", F_H: ", F_H, ", IN_C: ", IN_C)
        a = np.reshape(a, (OUT_C, F_W, F_H, IN_C)) 
        a = np.transpose(a, (1, 2, 3 ,0))
        w = a
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

  generate_tflite_v2(ITER, generate_method, target, w, model_name, outfile_name, ramdisk, paras, default_range = the_range, stats_values = stat_values)

#  print("test 1Kx1K random input result range.")
#  In = np.random.randint(0, 256, size=(1024, 1024))
#  We = np.random.randint(0, 256, size=(1024, 1024))
  
#  Out = np.matmul(In, We)
#  print("the output is:")
#  print(Out)
#  print("the average is: ", 128*128*1024)
#  print("the max is: ", Out.max(), ", min is: ", Out.min())
#  print("the middle is: ", (Out.max() + Out.min())/2)

#  scale = (Out.max() - Out.min()) / 255.0
#  mean  = Out.min()

#  sOut = (Out - mean) / scale
#  sOut = sOut.astype(np.uint8)
#  print("scale: ", scale, ", mean: ", mean )
#  print("the scaled output is:")
#  print(sOut)
#  print("the max is: ", sOut.max(), ", min is: ", sOut.min())
#  print("the middle is: ", (sOut.max() + sOut.min())/2)

#  rOut = (sOut * scale) + mean
#  print("the restored output is:")
#  print(rOut)
#  print("the max is: ", rOut.max(), ", min is: ", rOut.min())
#  print("the middle is: ", (rOut.max() + rOut.min())/2)

