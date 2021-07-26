import os
import numpy as np
import random

def save_weight(w, file_name):
  if os.path.isfile("../data/"+str(file_name)+".npy"):
    os.system("rm ../data/"+str(file_name)+".npy")
  print("write weight to file ../data/"+str(file_name)+" ...")
  np.save("../data/"+str(file_name), w)

def read_input(file_name, max_row):
  w = np.genfromtxt(file_name, usecols=0, max_rows = max_row)
  return w

def do_ref(pIn, tIn, m_size, layers, Cap, Rx, Ry, Rz, dt, num_layer):
# return tOut
  ce = 0.136533
  cw = 0.136533
  cn = 0.136533
  cs = 0.136533
  cc = 0.453667
  cb = 0.000067
  ct = 0.000067
  stepDivCap = 1.365333 

  tOut = np.zeros([m_size, m_size])

  for i in range(num_layer):
    print("num_layer: ", i)
    for y in range(m_size):
      for x in range(m_size):
        n_y = y if y ==0 else y-1
        s_y = y if y == m_size-1 else y+1
        e_x = x if x == m_size-1 else x+1
        w_x = x if x == 0 else x-1       
        tOut[y, x] = tIn[y, x] * (cc+ct+cb) + tIn[n_y, x]*cn + \
                                              tIn[s_y, x]*cs + \
                                              tIn[y, e_x]*ce + \
                                              tIn[y, w_x]*cw + \
                                              stepDivCap * pIn[y, x] + ct * amb_temp
#    print("during the conv+add: tOut max: ", tOut.max(), ", min: " , tOut.min(), ", mean: ", tOut.mean())
#    delete = stepDivCap * pIn + ct * amb_temp
#    print("during the conv+add: temp add max: ", delete.max(), ", min: " , delete.min(), ", mean: ", delete.mean())
  
#    for y in range(m_size):
#      for x in range(m_size):
#        tOut[y, x] += stepDivCap * pIn[y, x] + ct * amb_temp
 
    temp = tIn
    tIn = tOut
    tOut = temp
  return tOut                                      

def do_TPU(pIn, tIn, m_size, layers, Cap, Rx, Ry, Rz, dt, num_layer):
  ce = 0.136533
  cw = 0.136533
  cn = 0.136533
  cs = 0.136533
  cc = 0.453667
  cb = 0.000067
  ct = 0.000067
  stepDivCap = 1.365333 
  
  tIn_max_float = tIn.max()
  pIn =  stepDivCap * pIn + ct * amb_temp # modified for easy operating
  p_scale = pIn.max() / 255.0
  t_scale = tIn.max() / 255.0
  acc_scale = t_scale
  print(pIn)
  print("p: max:", pIn.max(), ", min: ", pIn.min(), ", mean: ", pIn.mean()) 
  print(tIn)
  print("t: max:", tIn.max(), ", min: ", tIn.min(), ", mean: ", tIn.mean()) 
 
  scales_list = []
  print("p_scale: ", p_scale, ", t_scale: ", t_scale)
  scales_list.append(p_scale)
  scales_list.append(t_scale)
  pIn = pIn / p_scale
  tIn = tIn / t_scale
  pIn = pIn.astype(np.uint8)
  tIn = tIn.astype(np.uint8)
  print(pIn, pIn.shape, "pIn mean: ", pIn.mean())  
  print(tIn, tIn.shape, "tIn mean: ", tIn.mean())  
  tOut = np.zeros([m_size, m_size])
  tOut = tOut.astype(np.uint8)

  cmax = max(cc+ct+cb, max(cn, max(cs, max(ce, cw))))
  icc = (int)(((cc+ct+cb) / cmax) * 255.0)
  icn = (int)((cn / cmax) * 255.0)
  ics = (int)((cs / cmax) * 255.0)
  ice = (int)((ce / cmax) * 255.0)
  icw = (int)((cw / cmax) * 255.0)

  print("scale: ", cmax/255.0 , "icc: ", icc, ", icn: ", icn, ", ics: ", ics, ", ice: ", ice, ", icw: ", icw)

  for i in range(num_layer):
    print("num_layer: ", i)
    tIn = tIn.astype(np.uint32)
    pIn = pIn.astype(np.uint32)
    tOut = tOut.astype(np.uint32)
    for y in range(m_size):
      for x in range(m_size):
        n_y = y if y ==0 else y-1
        s_y = y if y == m_size-1 else y+1
        e_x = x if x == m_size-1 else x+1
        w_x = x if x == 0 else x-1       
        tOut[y, x] = tIn[y, x] * icc + tIn[n_y, x]*icn + \
                                       tIn[s_y, x]*ics + \
                                       tIn[y, e_x]*ice + \
                                       tIn[y, w_x]*icw #+ \
#                                      stepDivCap * pIn[y, x] + ct * amb_temp
    it_scale = np.float32(tOut.max() / 255.0)  
    scales_list.append(np.float32(1.0/it_scale))
    print("i: ", i, ", after conv scale: ", scales_list[-1], ", tOut.max: ", tOut.max())
    acc_scale *= it_scale
    #ip_scale = 1.0 / (( (p_scale / t_scale) / it_scale))
    ip_scale = 1.0 / (( it_scale / p_scale))
    pIn_layer = ( pIn * ( 1.0 / ip_scale)).astype(np.uint32)
    scales_list.append(np.float32(1.0/ip_scale))
    print("i: ", i, ", add input scale: ", scales_list[-1])
# ====== add ====================
#    tOut += pIn_layer
    it_scale = np.float32(tOut.max() / 255.0)
    scales_list.append(np.float32(1.0/it_scale))
    print("i: ", i, ", after iter scale: ", scales_list[-1])
    acc_scale *= it_scale
    tOut = np.float32(tOut / it_scale).astype(np.uint8)
#    print("found scale for this layer: conv: ", scales_list[-3], ", add input scale: ", scales_list[-2], ", output scale: ", scales_list[-1])
#    print("tmp tOut.max: ", tOut.max(), ", min: ", tOut.min())
    temp = tIn
    tIn = tOut
    tOut = temp

  return tOut.astype(np.uint8), scales_list, acc_scale

if __name__ == "__main__":
  np.random.seed(9487)
  num_layer = 1
  m_size = 256
  max_row = m_size * m_size * 1
  pIn = read_input("../app/rodinia_3.1/data/hotspot3D/power_1024x1", max_row)
  tIn = read_input("../app/rodinia_3.1/data/hotspot3D/temp_1024x1", max_row)

  pIn = pIn.reshape((m_size, m_size))
  tIn = tIn.reshape((m_size, m_size))
  
  MAX_PD = 3.0e6
  PRECISION = 0.001#
  SPEC_HEAT_SI = 1.75e6
  K_SI = 100
  FACTOR_CHIP = 0.5
  t_chip = 0.0005
  chip_height = 0.016
  chip_width = 0.016
  amb_temp = 80.0

  dx = chip_height / m_size
  dy = chip_width / m_size
  dz = t_chip / 1
  Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip *dx * dy;
  Rx  = dy / (2.0 * K_SI * t_chip * dx)
  Ry  = dx / (2.0 * K_SI * t_chip * dy)
  Rz  = dz / (K_SI * dx * dy)
  max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI)
  dt = PRECISION / max_slope
# ===== do ref ====
  tOut_ref = do_ref(pIn, tIn, m_size, 1, Cap, Rx, Ry, Rz, dt, num_layer)
  print(tOut_ref, tOut_ref.shape, ", max: ", tOut_ref.max(), ", mean: ", tOut_ref.mean(), ", min: ", tOut_ref.min())
# ===== do mimiking tpu ====
  pIn = read_input("../app/rodinia_3.1/data/hotspot3D/power_1024x1", max_row)
  tIn = read_input("../app/rodinia_3.1/data/hotspot3D/temp_1024x1", max_row)

  pIn = pIn.reshape((m_size, m_size))
  tIn = tIn.reshape((m_size, m_size))

  tOut_TPU, scales_list, acc_scale = do_TPU(pIn, tIn, m_size, 1, Cap, Rx, Ry, Rz, dt, num_layer)
  #tOut_TPU = tOut_TPU * acc_scale
  range_ref = tOut_ref.max() - tOut_ref.min()
  mean_ref  = tOut_ref.mean()
  range_TPU = tOut_TPU.max() - tOut_TPU.min()
  mean_TPU   = tOut_TPU.mean()

  tOut_TPU = tOut_TPU.astype(np.int32)
  tOut_TPU = tOut_TPU - mean_TPU
  tOut_TPU = (tOut_TPU / range_TPU ) * range_ref
  tOut_TPU = tOut_TPU + mean_ref 

  print(scales_list)
  print(tOut_TPU, tOut_TPU.shape, ", max: ", tOut_TPU.max(), ", mean: ", tOut_TPU.mean(), ", min: ", tOut_TPU.min())
#                                   t_scale 
  print("accumulated scale: ", acc_scale)
# ===== measurement report =====
  rmse = np.sqrt(np.mean((tOut_TPU - tOut_ref)**2))
  error = np.mean(np.abs(tOut_TPU - tOut_ref))

  print("rmse: ", rmse, ", rmse%: ", (rmse / tOut_ref.mean())*100, " #%"     )
  print("error: ", error, ", error%: ", (error / tOut_ref.mean())*100, " %"     )
