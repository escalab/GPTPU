# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.keras.models import * #Model
from tensorflow.python.keras.layers import * #Input, Reshape, Dense, Add, Conv2D, ZeroPadding2D, AveragePooling2D
import os
import math
import argparse

# Model parameters
# MM_M = (W/(F_W*P_W))*(H/(F_H*P_H))
# MM_N = (F_W*P_W)*(F_H*P_H)*IN_C
# MM_K = OUT_C
parser = argparse.ArgumentParser()
parser.add_argument("--m", action="store", dest="m", default=1024)
parser.add_argument("--n", action="store", dest="n", default=1024)
parser.add_argument("--k", action="store", dest="k", default=1024)
parser.add_argument("--iter", action="store", dest="iter", default=1, help="iteration for gptpu_imv()")
parser.add_argument("--mode", action="store", dest="mode", default=0, help="mode 0 for mm2conv; mode 1 for 256mmOnConv(16b->32b); mode=2 for 256mmConv(8b->32b)")
args = parser.parse_args()
MM_M = int(args.m)
MM_N = int(args.n)
MM_K = int(args.k)
Iter = int(args.iter)
mode = int(args.mode)
N = max(max(MM_N, MM_N), MM_K)
n = int(math.log2(N))


par = [ 2**j for j in range(1, (n if mode == 0 else 14)+1)] # 2**14 = 4096 = 256 * 16
p_par = [2**j for j in range(0, 5)] # upto 16
f_par = [2**j for j in range(0, 7)] # upto 64


for fold in f_par:
  for P_W in p_par:
    for P_H in p_par:
      if Iter > 1:
        model_shape_name = "model_convAvg_"+str(MM_M)+"x"+str(MM_N)+"x"+str(MM_K)+"_P"+str(P_W)+"x"+str(P_H)+"_fold"+str(fold)+"_iter"+str(Iter)
      else:
        model_shape_name = "model_convAvg_"+str(MM_M)+"x"+str(MM_N)+"x"+str(MM_K)+"_P"+str(P_W)+"x"+str(P_H)+"_fold"+str(fold)
               
      os.system("rm -f ./"+model_shape_name+"/*")
      os.system("mkdir -p ./"+model_shape_name)
      
      for INTERNAL_MODEL_WIDTH in par:
        for INTERNAL_MODEL_HEIGHT in par:
          for MODEL_CHANNEL in par:
            for FILTER_KERNEL_W in par:
              for FILTER_KERNEL_H in par:
                for FILTER_CHANNEL in ([MM_K] if mode == 0 else reversed(par)):
                  print(INTERNAL_MODEL_WIDTH, INTERNAL_MODEL_HEIGHT, MODEL_CHANNEL, FILTER_KERNEL_W, FILTER_KERNEL_H, P_W, P_H, fold)          
                  if (mode == 0 and 
                        MM_M == int((INTERNAL_MODEL_WIDTH/(FILTER_KERNEL_W*P_W))*(INTERNAL_MODEL_HEIGHT/(FILTER_KERNEL_H*P_H))) and 
                        MM_N == (FILTER_KERNEL_W*P_W)*(FILTER_KERNEL_H*P_H)*MODEL_CHANNEL and 
                        INTERNAL_MODEL_WIDTH >= FILTER_KERNEL_W and 
                        INTERNAL_MODEL_HEIGHT >= FILTER_KERNEL_H) \
                    or (mode == 1 and 
                        FILTER_KERNEL_W * FILTER_KERNEL_H *  MODEL_CHANNEL == 256 and 
                      int((INTERNAL_MODEL_WIDTH/FILTER_KERNEL_W)*(INTERNAL_MODEL_HEIGHT/FILTER_KERNEL_H)) == 4096 and 
                      INTERNAL_MODEL_WIDTH >= FILTER_KERNEL_W and 
                      INTERNAL_MODEL_HEIGHT >= FILTER_KERNEL_H) \
                  or (mode == 2 and 
                      FILTER_KERNEL_W * FILTER_KERNEL_H * MODEL_CHANNEL == 256 and 
                      int((INTERNAL_MODEL_WIDTH/FILTER_KERNEL_W)*(INTERNAL_MODEL_HEIGHT/FILTER_KERNEL_H)) == 2048 and 
                      INTERNAL_MODEL_WIDTH >= FILTER_KERNEL_W and INTERNAL_MODEL_HEIGHT >= FILTER_KERNEL_H ):
                    STRIDE_W = FILTER_KERNEL_W
                    STRIDE_H = FILTER_KERNEL_H
                    FILTER_LAYERS_NUM = Iter
                  # Create model structure
                    print("IN_W="+str(INTERNAL_MODEL_WIDTH)+", IN_H="+str(INTERNAL_MODEL_HEIGHT)+", IN_C="+str(MODEL_CHANNEL)+", layers="+str(FILTER_LAYERS_NUM)+", OUT_C="+str(FILTER_CHANNEL)+", F_W="+str(FILTER_KERNEL_W)+", F_H="+str(FILTER_KERNEL_H)+", P_W="+str(P_W)+", P_H="+str(P_H))
                    input0 = Input(shape=(INTERNAL_MODEL_WIDTH, INTERNAL_MODEL_HEIGHT, MODEL_CHANNEL*fold))
                    conv = input0
                    if fold == 1:
                      for i in range(FILTER_LAYERS_NUM):
                        conv = Conv2D(
                          filters=FILTER_CHANNEL,
                          kernel_size=(FILTER_KERNEL_W,FILTER_KERNEL_H),
                          strides=(STRIDE_W, STRIDE_H),
                   #      padding='same',
                          activation='linear'
                        )(conv)
                      conv = AveragePooling2D(pool_size=(P_W, P_H), strides=(P_W, P_H))(conv)
                      model = Model(inputs=[input0], outputs=[conv])
                    else: # fold > 1
                      split = Lambda(lambda x: tf.split(x, num_or_size_splits=fold, axis=3))(conv)
                      conv_list = []
                      for i in range(fold):
                        conv = split[i]
                        for i in range(FILTER_LAYERS_NUM):
                          conv = Conv2D(
                            filters=FILTER_CHANNEL,
                            kernel_size=(FILTER_KERNEL_W,FILTER_KERNEL_H),
                            strides=(STRIDE_W, STRIDE_H),
                     #      padding='same',
                            activation='linear'
                          )(conv)
                        conv = AveragePooling2D(pool_size=(P_W, P_H), strides=(P_W, P_H))(conv)
                        conv_list.append(conv)
                      out = concatenate(conv_list, axis=3)
                      model = Model(inputs=[input0], outputs=out)
                  # Save model
                    model.summary()
                    model.compile(
                      optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy']
                    )

                    model_name = './'+model_shape_name+'/model_conv_' + str('{0:04}'.format(INTERNAL_MODEL_WIDTH)) +'x' + str('{0:04}'.format(INTERNAL_MODEL_HEIGHT)) + 'x' + str('{0:04}'.format(MODEL_CHANNEL)) +'x' + str('{0:04}'.format(FILTER_CHANNEL)) +'x' + str('{0:04}'.format(FILTER_KERNEL_W)) + 'x' + str('{0:04}'.format(FILTER_KERNEL_H)) +'x' + str('{0:04}'.format(STRIDE_W)+'x'+str('{0:04}'.format(STRIDE_H)))+'x'+str('{0:04}'.format(Iter))

                    model.save(model_name + '.h5')

      # Convert to quantized tflite
                    converter = tf.lite.TFLiteConverter.from_keras_model_file(model_name + '.h5')
                    converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
                    converter.default_ranges_stats = (0, 6)
                    input_arrays = converter.get_input_arrays()
                    converter.quantized_input_stats = {input_arrays[0] : (128., 127.)}  # mean, std_dev
                    tflite_model = converter.convert()
                    open(model_name + '.tflite', "wb").write(tflite_model)
          
                    #os.system("timeout 5 edgetpu_compiler "+model_name+".tflite -o ./"+model_shape_name+"/ -s")
                    print(model_name)
                    print("timeout 60 edgetpu_compiler "+model_name+".tflite -o ./"+model_shape_name+"/ -s")
                    os.system("timeout 60 edgetpu_compiler "+model_name+".tflite -o ./"+model_shape_name+"/ -s")
                    #MM_M = int((INTERNAL_MODEL_WIDTH/FILTER_KERNEL_W)*(INTERNAL_MODEL_HEIGHT/FILTER_KERNEL_H))
                    #MM_N = FILTER_KERNEL_W*FILTER_KERNEL_H*MODEL_CHANNEL
                    #MM_K = FILTER_CHANNEL
                    #print("mapped mm shape: ("+str(MM_M)+"x"+str(MM_N)+"x"+str(MM_K)+")") if mode == 0 else print("256mm block done")
