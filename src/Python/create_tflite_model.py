import os
import os.path
import shutil
import glob
import argparse
import numpy as np
import tensorflow as tf
from models import Models
from utils.utils import (
        get_saved_model_path_base,
        remove_files_except
        )
import tensorflow.keras as keras

def try_numerical(params):
    """ integer prioritized type casting function. """
    for i, param in enumerate(params):
        try:
            params[i] = int(param)
        except ValueError:
            params[i] = str(param)
    return params

def specify_model_weights(model, model_name, params, weight_file = ""):
    """ A generic way to specify model weights including models that have \
        multiple layers that cannot be fully initialized by giving weights \
        during model creation(model.py stage). """

    if model_name == "conv2d" and weight_file != "":
        [w, h, in_c, out_c, f_w, f_h, s_w, s_h, padding] = params
        with open(weight_file, "rb") as f:
            w = np.fromfile(f, dtype=np.float32)
            w = np.reshape(w, (out_c, f_w, f_h, in_c))
            w = np.transpose(w, (1, 2, 3, 0))
            new_weights = [w]
    else:
        old_weights = model.get_weights()
        new_weights = []
        for w in old_weights:
            # currently gives all ones
            #per_layer_weights = np.ones(w.shape)
            per_layer_weights = np.random.randint(0, 256, w.shape)
            new_weights.append(per_layer_weights)
    model.set_weights(new_weights)

def main(args):
    # setup
    params     = try_numerical(args.params)
    in_shape   = tuple(try_numerical(args.in_shape))
    name_postfix = '-'.join([str(i) for i in params])
    model_name = (str(args.model) + "_" + name_postfix)
    frozen_model_path       = os.path.join(args.saved_model_base, model_name, ".pb")
    saved_model_path        = os.path.join(args.saved_model_base, model_name)
    xxx_edgetpu_tflite_name = model_name + "_edgetpu.tflite"
    xxx_edgetpu_tflite_path = os.path.join(saved_model_path, xxx_edgetpu_tflite_name) 
    os.system("mkdir -p -m 777 "+saved_model_path)

    # frozen the model (store to 'frozen_model_path')
    eval_graph = tf.Graph()
    eval_sess = tf.compat.v1.Session(graph=eval_graph)
    keras.backend.set_session(eval_sess)

    with eval_graph.as_default():
        keras.backend.set_learning_phase(0)
        # get the model
        model = Models(args.model).get_model(args.params)
        model.summary()
        in_layer_name  = model.layers[0].input.name.split(':')[0]
        out_layer_name = model.layers[-1].output.name.split(':')[0]
        tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
        eval_graph_def = eval_graph.as_graph_def()
        # To initailize all varaibles
        init = tf.initialize_all_variables()
        eval_sess.run(init)
        # set custom weights to an arbitary model
        specify_model_weights(model, args.model, args.params, args.weight_file_path)
        # check weights
        print(model.get_weights())
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            eval_sess,
	    eval_graph_def,
            [model.output.op.name]
        )
    with open(frozen_model_path, 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())

    # convert tf model to tflite model using toco convereter (tf1)
    model_path = os.path.join(saved_model_path, model_name+".tflite")
    os.system("rm -f "+model_path) # Make sure old one is not used.  
    cmd = "toco --output_file="+   model_path + \
	  " --graph_def_file="+    frozen_model_path + \
 	  " --input_array="+       in_layer_name + \
	  " --output_array="+      out_layer_name + \
	  " --inference_type="+    "QUANTIZED_UINT8" + \
	  " --default_ranges_min="+str(args.range_min) + \
	  " --default_ranges_max="+str(args.range_max) + \
	  " --mean_value="+        str(args.mean) + \
	  " --std_dev_values="+    str(1.0/(float)(args.scale))
    os.system(cmd)

    # convert xxx.tflite to xxx_edgetpu.tflite
    cmd = ("edgetpu_compiler -s -m 13 "+model_path+" -o "+saved_model_path)
    print(cmd)
    os.system(cmd)

    # convert xxx_edgetpu.tflite binary to to json
    if(not os.path.exists("schema_v3.fbs")): # download schema if not present
        os.system("wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/schema/schema_v3.fbs")
    cmd = ("flatc -t -o "+saved_model_path+" --strict-json --defaults-json schema_v3.fbs \
               -- "+xxx_edgetpu_tflite_path)
    print(cmd)
    os.system(cmd)

    # remove all generated file except xxx_edgetpu.tflite
    print("saved_model_path: ", saved_model_path)
    if args.save_edgetpu_tflite_only == True:
        remove_files_except(saved_model_path, "*_edgetpu.tflite")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script generates tflite model for running on edgeTPU.")
    parser.add_argument('--model', action='store', type=str, required=True, help='model/op name')
    parser.add_argument('--in_shape', nargs="*", required=True, help="Required input tensor shape for chosen model. Ex: --in_shape 1024 1024")
    parser.add_argument('--params', nargs="*", required=True, help="Required params list for chosen model. conv2d as example: --params 1024 1024 1 1 1 1 1 1 valid")
    parser.add_argument('--saved_model_base', action='store', type=str, help='saved models base path')
    parser.add_argument('--save_edgetpu_tflite_only', action='store_true', help='skip saving all intermediate/postprocess files except the final xxx_edgetpu.tflite. (i.e.: tf model/xxx.tflite/*.json are skipped)')
    parser.add_argument('--range_min', action='store', type=float, help=' toco convert min value')
    parser.add_argument('--range_max', action='store', type=float, help=' toco convert max value')
    parser.add_argument('--mean', action='store', type=float, help=' toco convert mean value')
    parser.add_argument('--scale', action='store', type=float, help=' toco convert scale value')
    parser.add_argument('--weight_file_path', action='store', type=str, help='use weight file to specify model\'s weight. (If applicable)')

    # A default path base aligned with dir structure design.
    parser.set_defaults(saved_model_base=get_saved_model_path_base())
    parser.set_defaults(save_edgetpu_tflite_only=False)
    parser.set_defaults(range_min=0)
    parser.set_defaults(range_max=255)
    parser.set_defaults(mean=0)
    parser.set_defaults(scale=1)
    parser.set_defaults(weight_file_path="")

    args = parser.parse_args()
    print(args)
    main(args)
