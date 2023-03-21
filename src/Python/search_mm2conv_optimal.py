import os
import math
import tensorflow.keras
import argparse
import tensorflow as tf
from tensorflow.keras import layers
from utils.utils import (
        get_gittop,
        get_saved_model_path_base)

def is_valid_mapping(M, N, K, IN_W, IN_H, IN_C, F_W, F_H, OUT_C):
    """ a valid mm2conv shape mapping checker. """
    # valid rules: 
    # M = (IN_W / F_W) * (IN_H / F_H)
    # N = F_W * F_H * IN_C
    # K = OUT_C
    return True if (IN_W >= F_W and \
                    IN_H >= F_H and \
                    IN_W % F_W == 0 and \
                    IN_H % F_H == 0 and \
                    ((IN_W / F_W) * (IN_H / F_H)) == M and \
                    (F_W * F_H * IN_C) == N and \
                    OUT_C == K) else False

def params_gen(par):
    """ A yielding generator that goes through all valid params. """
    for IN_W in par:
        for IN_H in par:
            for IN_C in par:
                for F_W in par:
                    for F_H in par:
                        for OUT_C in par:
                            if is_valid_mapping(M, N, K, IN_W, IN_H, IN_C, F_W, F_H, OUT_C):
                                yield (IN_W, IN_H, IN_C, F_W, F_H, OUT_C)

def main(M, N, K, iters):
    # searching space setup
    max_dim = max(max(M, N), K)
    twos_power_max = int(math.log2(max_dim))
    twos_powers = [2**j for j in range(1, twos_power_max+1)]

    print(twos_powers)

    # generate candidates
    for params in params_gen(twos_powers):
        IN_W, IN_H, IN_C, F_W, F_H, OUT_C = params
        M = (IN_W / F_W) * (IN_H / F_H)
        N = F_W * F_H * IN_C
        K = OUT_C
        print(params)
        os.system("python create_tflite_model.py --model conv2d --in_shape "+ \
                  str(IN_W)+" "+str(IN_H)+ \
                  " --params "+ \
                  str(IN_W) + " " + \
                  str(IN_H) + " " + \
                  str(IN_C) + " " + \
                  str(OUT_C) + " " + \
                  str(F_W) + " " + \
                  str(F_H) + " " + \
                  str(F_W) + " " + \
                  str(F_H) + " " + \
                  "same --edgetpu_tflite_only")

        # run a candidate
        saved_model_base = get_saved_model_path_base()
        params = [IN_W, IN_H, IN_C, OUT_C, F_W, F_H, F_W, F_H, "same"]
        name_postfix = '-'.join([str(i) for i in params])
        model_name = "conv2d_" + name_postfix
        xxx_edgetpu_tflite_path = os.path.join(get_gittop(), "kernels", "templates", model_name, model_name+"_edgetpu.tflite")
        cmd = args.model_run_path + " " + \
            xxx_edgetpu_tflite_path + " " + \
            str(IN_W*IN_H) + " " + \
            str(IN_W*IN_H) + " " + str(iters) + \
            " 1"; # enable_log

        if os.path.exists(xxx_edgetpu_tflite_path):
            os.system(cmd)

        # delete the model now to save space during exhaustic searching

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", action="store", required=True, \
                        help="[M] n k  dimension of a GEMM")
    parser.add_argument("--n", action="store", required=True, \
                        help=" m [N] k dimension of a GEMM")
    parser.add_argument("--k", action="store", required=True, \
                        help=" m n [K] dimension of a GEMM")
    parser.add_argument("--iter", action="store", \
                        help="# of iteration of per invoke, for averaginf purpose.")
    parser.add_argument("--model_run_path", action="store", type=str, \
                        help="The executable path that can run a arbitary \
                        model with timing result.")
    parser.set_defaults(model_run_path=os.path.join(get_gittop(), \
                                                    "out", \
                                                    "k8", \
                                                    "minimal"))
    parser.set_defaults(iters=100)
    args = parser.parse_args()

    M = int(args.m)
    N = int(args.n)
    K = int(args.k)
    iters = int(args.iters)

    main(M, N, K, iters)
