import os
import configparser
import numpy as np

# should be deprecated in the future
sd = 0 # input vector digits separation

def creat_arrays(file_path):
  with open(file_path, "r") as f:
    data = f.read().split()
    floats = []
    for a in data:
      floats.append(float(a)) 
  return np.asarray(floats)

def get_model_output(model, sd, tid, verbose):
  os.system(" ../edgetpu/out/k8/examples/dense_minimal "+str(tid)+" 1 "+str(verbose)+" ../data/"+str(model)+"_tflite/"+str(model)+"_quant.tflite "+str(sd))
#  print("run.py: get_model_output\n")
  return creat_arrays("../data/output.txt") 

def run(sd, sw, tid, verbose):
  # creat model graph
  
#  os.system("python3 ./../src/create_model.py")
  # run on device
  outputs = get_model_output("test_model", sd, tid, verbose)
  inputs  = creat_arrays("../data/input.txt")
  w = np.load("../data/weight.npy", allow_pickle=True)
  if sw == 1:
    outputs_1th = get_model_output("test_model_1th_digit", sd, tid, verbose)
    w2 = np.load("../data/weight_1st.npy", allow_pickle=True)
    outputs_2th = get_model_output("test_model_2th_digit", sd, tid, verbose)
    w2 = np.load("../data/weight_2nd.npy", allow_pickle=True)
    out = (inputs, outputs, w, outputs_1th, outputs_2th)
  else:
    out = (inputs, outputs, w)
  return out

def error_calculate(inputs, outputs, w):
  precise_out = np.matmul(inputs, w)
  overflow_mask = [False if (x > 255) else True for x in precise_out ]
  overflows = len(precise_out) - np.sum(overflow_mask)
  #print("# of overflow positoin(s): ", overflows)

  # stats
  # all percentage
  percent = (abs(outputs - precise_out) / precise_out)*100
  #print("all error percentage: ", percent)
  avg_percent = sum(percent) / len(percent)  

  # percentage on non-overflow position(s)
  non_overflow_percent = percent[overflow_mask]
  #print("non-overflow error %: ", non_overflow_percent)
  avg_non_overflow_percent = sum(non_overflow_percent) / len(non_overflow_percent) if (len(non_overflow_percent) > 0) else 0
 
  return len(precise_out), len(precise_out) - overflows, avg_percent, avg_non_overflow_percent

def accumulate_stats(accu, curr):
  accu_t_cnt, accu_n_cnt, accu_t_avg, accu_n_avg = accu
  totals, non_overflows, all_avg, non_avg = curr  
  print(totals, non_overflows, all_avg, non_avg)
  accu_t_avg = ((accu_t_avg * accu_t_cnt) + (all_avg * totals))/(accu_t_cnt + totals) if (accu_t_cnt + totals) > 0 else 0
  accu_n_avg = ((accu_n_avg * accu_n_cnt) + (non_avg * non_overflows))/(accu_n_cnt + non_overflows) if (accu_n_cnt + non_overflows) > 0 else 0
  accu_n_cnt += non_overflows
  accu_t_cnt += totals
  return accu_t_cnt, accu_n_cnt, accu_t_avg, accu_n_avg

def error_and_stats(data, accu):
  # de tuple
  inputs, outputs, w = data     
  # ===== start =====
  curr = error_calculate(inputs, outputs, w)   
  accu = accumulate_stats(accu, curr)  
  return accu

if __name__ == "__main__":
  config = configparser.ConfigParser()
  config.read('../config.ini')  
  sw      = int(config['FLAG']['sep'])
  n       = int(config['FLAG']['iter'])
  tid     = int(config['SYSTEM']['tid'])
  verbose = int(config['SYSTEM']['verbose'])

  len_stats = 2 if sw == 1 else 1
  accu_t_cnt = [0] * len_stats
  accu_n_cnt = [0] * len_stats
  accu_t_avg = [0] * len_stats
  accu_n_avg = [0] * len_stats
  for i in range(n):
    out = run(sd, sw, tid, verbose)  
    data = out[:3] if sw == 1 else out

    for i in range(len_stats):
      if i == 1: # only sw is enabled would touch this
        inputs, outputs, w, outputs_1th, outputs_2th = out
        outputs2 = np.add(outputs_1th, outputs_2th/10) 
        print("inputs: ", inputs, ", w: ", w)
        print("precise_output: ", np.matmul(inputs, w), ", output1: ", outputs, ", outputs2: ", outputs2, ", outputs_1th: ", outputs_1th, ", outputs_2th: ", outputs_2th)
        data = (inputs, outputs2, w)
      accu = ( accu_t_cnt[i], accu_n_cnt[i], accu_t_avg[i], accu_n_avg[i] )
      accu_t_cnt[i], accu_n_cnt[i], accu_t_avg[i], accu_n_avg[i] = error_and_stats(data, accu)

  print("===== report =====")
  print("# of iterations: ", n)
  for i in range(len_stats):
    print("accu ", i , ":")
    print("# of accumulatived elements so far: ", accu_t_cnt[i])
    print("# of non-overflow  elements so far: ", accu_n_cnt[i])
    print("average error percentage of all elements         : ", accu_t_avg[i])
    print("average error percentage of non-overflow elements: ", accu_n_avg[i])
    

