import os
import numpy as np
import random

def save_weight(w, file_name):
  if os.path.isfile("../data/"+str(file_name)+".npy"):
    os.system("rm ../data/"+str(file_name)+".npy")
  print("write weight to file ../data/"+str(file_name)+" ...")
  np.save("../data/"+str(file_name), w)

if __name__ == "__main__":
  np.random.seed(9487)
  num_layer = 5
  m_size = 1024
  pIn = np.random.randint(1, 2, size=(m_size))
  pIn = (pIn.astype(np.float32) / m_size) # all are initialzed as 1/size
  pWe = np.random.randint(0, 2, size=(m_size, m_size)) # either 0 or 1
  pWe = pWe.astype(np.float32)
#  pWe = np.swapaxes(pWe, 0, 1)
  print(pWe)
  save_weight(pWe, "pagerank_1K_iter5_weight")

  min_n = m_size
  for i in range(m_size):
    n = (pWe[:,i] == 1).sum() 
    min_n = min(min_n, n)
    pWe[:,i] = pWe[:,i] / n
  print("weight float is:")
  print(pWe)  

  oOut = pIn
  for i in range(num_layer):
    oOut = np.matmul(pWe, oOut)
  print("float reference output is:")
  print(oOut)

  sIn_scale = 255.0 * m_size
  sWe_scale = 255.0 * min_n

  sIn = pIn * sIn_scale
  sIn = sIn.astype(np.uint8)
  sWe = pWe * sWe_scale
  sWe = sWe.astype(np.uint8)

  print("sIn: ", sIn)
  print("sWe: ", sWe)

  def tpu_mul(x, y):
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    res = np.matmul(x, y)
#    scale = (res.max() - res.min()) / 255.0
#    mean  = res.min()
    scale = np.float32(res.max() / 255.0)
    mean  = 0.0

    sOut = np.float32(((res - mean) / scale)).astype(np.uint8)
    print("found scale for this layer: ", np.float32(1.0/scale))
    print("tmp sOut.max ", sOut.max(), ", min: ", sOut.min())
    return (sOut, scale, mean)

  sOut = sIn
  scale_list = []
  mean_list  = []
  for i in range(num_layer):
    (sOut, scale, mean) = tpu_mul(sWe, sOut)
# restored sOut should be in float: sOut = (sOut * scale) + mean
    scale_list.append(scale)
    mean_list.append(mean)
#    print("max of tmp sOut: ", sOut.max(), ", min: ", sOut.min())
#    print("restored sOut in float:")
#    print((sOut * scale) + mean )

  for i in range(num_layer):
    sOut = (sOut * scale_list[i]) + mean_list[i]
#    sOut = sOut / (sIn_scale * sWe_scale)
  
  print("sOut before scaling is:")
  print(sOut)
  sum_scale = sOut.sum()
  print(sum_scale)
  sOut = sOut / sum_scale

  print("sOut is:")
  print(sOut)
  print("scales: ")
  print(scale_list)
  print("means: ")
  print(mean_list)

  rmse = np.sqrt(np.mean((oOut-sOut)**2))#
  error = np.mean(np.abs(oOut - sOut))

  print("rmse: ", rmse, ", rmse%: ", (rmse / oOut.mean())*100, " %"     )
  print("error: ", error, ", error%: ", (error / oOut.mean())*100, " %"     )
