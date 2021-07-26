# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Reshape, Dense, Add, Conv2D, ZeroPadding2D
import os
import math
import argparse

# Model parameters
# MM_M = (W/F_W)*(H/F_H)
# MM_N = F_W*F_H*IN_C
# MM_K = OUT_C
parser = argparse.ArgumentParser()
parser.add_argument("--m", action="store", dest="m", default=2048)
parser.add_argument("--n", action="store", dest="n", default=2048)
parser.add_argument("--k", action="store", dest="k", default=2048)
parser.add_argument("--iter", action="store", dest="iter", default=1, help="iteration for gptpu_imv()")
args = parser.parse_args()
MM_M = int(args.m)
MM_N = int(args.n)
MM_K = int(args.k)
Iter = int(args.iter)
N = max(max(MM_N, MM_N), MM_K)
n = int(math.log2(N))

par = [ 2**j for j in range(1, n+1)] # 2**14 = 4096 = 256 * 16

for acc in [4096]: #par: # a re-order to search throught sizes from small acc to larger acc
  os.system("rm -f ./all_conv_models/acc_"+str(acc)+"/*")
  os.system("mkdir -p ./all_conv_models/acc_"+str(acc))
  for INTERNAL_MODEL_WIDTH in par:
    for INTERNAL_MODEL_HEIGHT in par:
      for MODEL_CHANNEL in par:
        for FILTER_KERNEL_W in par:
          for FILTER_KERNEL_H in par:
            for FILTER_CHANNEL in par:
              print(INTERNAL_MODEL_WIDTH, INTERNAL_MODEL_HEIGHT, MODEL_CHANNEL, FILTER_KERNEL_W, FILTER_KERNEL_H, FILTER_CHANNEL)          
              if INTERNAL_MODEL_WIDTH >= FILTER_KERNEL_W and INTERNAL_MODEL_HEIGHT >= FILTER_KERNEL_H and (FILTER_KERNEL_W * FILTER_KERNEL_H * MODEL_CHANNEL == acc):
                STRIDE_W = FILTER_KERNEL_W
                STRIDE_H = FILTER_KERNEL_H
                FILTER_LAYERS_NUM = Iter
# Create model structure
                print("IN_W="+str(INTERNAL_MODEL_WIDTH)+", IN_H="+str(INTERNAL_MODEL_HEIGHT)+", IN_C="+str(MODEL_CHANNEL)+", layers="+str(FILTER_LAYERS_NUM)+", OUT_C="+str(FILTER_CHANNEL)+", F_W="+str(FILTER_KERNEL_W)+", F_H="+str(FILTER_KERNEL_H))
                input0 = Input(shape=(INTERNAL_MODEL_WIDTH,INTERNAL_MODEL_HEIGHT,MODEL_CHANNEL))
                conv = input0
                for i in range(FILTER_LAYERS_NUM):
                  conv = Conv2D(
                    filters=FILTER_CHANNEL,
                    kernel_size=(FILTER_KERNEL_W,FILTER_KERNEL_H),
                    strides=(STRIDE_W, STRIDE_H),
#      padding='same',
                    activation='linear'
                  )(conv)
                model = Model(inputs=[input0], outputs=[conv])
# Save model
                model.summary()
                model.compile(
                  optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                )

                model_name = './all_conv_models/acc_'+str(acc)+'/model_conv_' + str('{0:04}'.format(INTERNAL_MODEL_WIDTH)) +'x' + str('{0:04}'.format(INTERNAL_MODEL_HEIGHT)) + 'x' + str('{0:04}'.format(MODEL_CHANNEL)) +'x' + str('{0:04}'.format(FILTER_CHANNEL)) +'x' + str('{0:04}'.format(FILTER_KERNEL_W)) + 'x' + str('{0:04}'.format(FILTER_KERNEL_H)) +'x' + str('{0:04}'.format(STRIDE_W)+'x'+str('{0:04}'.format(STRIDE_H)))+'x'+str('{0:04}'.format(Iter))

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
              #print("timeout 60 edgetpu_compiler "+model_name+".tflite -o ./"+model_shape_name+"/ -s")
              #os.system("timeout 60 edgetpu_compiler "+model_name+".tflite -o ./"+model_shape_name+"/ -s")
              #MM_M = int((INTERNAL_MODEL_WIDTH/FILTER_KERNEL_W)*(INTERNAL_MODEL_HEIGHT/FILTER_KERNEL_H))
              #MM_N = FILTER_KERNEL_W*FILTER_KERNEL_H*MODEL_CHANNEL
              #MM_K = FILTER_CHANNEL
              #print("mapped mm shape: ("+str(MM_M)+"x"+str(MM_N)+"x"+str(MM_K)+")") if mode == 0 else print("256mm block done")
