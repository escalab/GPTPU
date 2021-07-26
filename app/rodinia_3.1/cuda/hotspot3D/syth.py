In  = '../../data/hotspot3D/power_8192x8'
Out = '../../data/hotspot3D/power_16384x8'

with open(Out, 'wb') as o:
  for idx in range(4):
    with open(In, 'rb') as i:
      for line in i.readlines():
        o.write(line)


