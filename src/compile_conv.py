# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Reshape, Dense, Add, Conv2D, ZeroPadding2D
import glob
import os
import math
import argparse

# Model parameters
# MM_M = (W/F_W)*(H/F_H)
# MM_N = F_W*F_H*IN_C
# MM_K = OUT_C
parser = argparse.ArgumentParser()
parser.add_argument("--in_path",  action="store", dest="in_path",  default=1, help="xxx_quant.tflite path")
parser.add_argument("--out_path", action="store", dest="out_path", default=1, help="path for xxx_quant_edgetpu.tflite")
args = parser.parse_args()
in_path  = args.in_path
out_path = args.out_path

names = glob.glob(in_path+"/*.tflite")

os.system("rm -rf "+out_path)
os.system("mkdir -p "+out_path)
print("begining of compile_conv.py...")
for name in names:  
        print(    "timeout 300 edgetpu_compiler "+name+" -o "+out_path+"/ -s")
        os.system("timeout 300 edgetpu_compiler "+name+" -o "+out_path+"/ -s")
        os.system("echo $?")# print exit status of COMMAND
