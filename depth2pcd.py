import argparse
import numpy as np
import inspect
import scipy.misc
from decimal import Decimal

parser = argparse.ArgumentParser(description='Numpy depth maps of KITTI dataset to PCD format')
parser.add_argument("-i", "--depth_file", required=True, help="name of the user")
parser.add_argument("-o", "--out_file", required=True)

opt = parser.parse_args()

def save_point_cloud(filename, pc):
  header = "VERSION .7\n" \
  "FIELDS x y z\n" \
  "SIZE 8 8 8\n" \
  "TYPE F F F\n" \
  "COUNT 1 1 1\n" \
  "WIDTH {}\n" \
  "HEIGHT 1\n" \
  "VIEWPOINT 0 0 0 1 0 0 0\n" \
  "POINTS {}\n" \
  "DATA ascii\n".format(len(pts), len(pts))

  fout = open(filename, 'w')
  fout.write(header)
  for i in range(len(pts)):
    fout.write('{:E} {:E} {:E}\n'.format(Decimal(pts[i][0]), Decimal(pts[i][1]), Decimal(pts[i][2])))
  fout.close()

depth_maps = np.load(opt.depth_file)
num_seq = 5
for k in range(num_seq):
  dimg = depth_maps[k]
  pts = []
  ratio = dimg.shape[0]*1.0/dimg.shape[1]
  for i in range(dimg.shape[0]):
    for j in range(dimg.shape[1]):
      pts.append([ratio*i*1.0/dimg.shape[0] - 0.5, j*1.0/dimg.shape[1] - 0.5, dimg[i][j]])

  min_depth, max_depth = np.min(dimg), np.max(dimg)
  pts = np.array(pts)
  pts[:,0] *= (max_depth - min_depth)
  pts[:,1] *= (max_depth - min_depth)

  # Save pcd
  save_point_cloud(opt.out_file.split('.')[0] + "-{}.pcd".format(k+1), pts)
  scipy.misc.imsave(opt.out_file.split('.')[0] + "-{}.png".format(k+1), dimg)
