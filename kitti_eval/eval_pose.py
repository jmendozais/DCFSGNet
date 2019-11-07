# Mostly based on the code written by Tinghui Zhou: 
# https://github.com/tinghuiz/SfMLearner/blob/master/kitti_eval/eval_pose.py
from __future__ import division
import os
import numpy as np
import argparse
from glob import glob
from pose_evaluation_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--gtruth_dir", type=str, 
    help='Path to the directory with ground-truth trajectories')
parser.add_argument("--pred_dir", type=str, 
    help="Path to the directory with predicted trajectories")
parser.add_argument('--iter', type=int, default=-1, help="Iteration of the model")
args = parser.parse_args()

def main():
    pred_files = glob(args.pred_dir + '/*.txt')
    ate_all = []
    for i in range(len(pred_files)):
        gtruth_file = args.gtruth_dir + os.path.basename(pred_files[i])
        if not os.path.exists(gtruth_file):
            continue
        ate = compute_ate(gtruth_file, pred_files[i])
        if ate == False:
            continue
        ate_all.append(ate)
    ate_all = np.array(ate_all)
    te_all, re_all = eval_odometry_kitti(args.gtruth_dir, args.pred_dir)
    limits = np.array(list(range(1,9)))*25
    te_all_2, re_all_2 = eval_odometry_kitti(args.gtruth_dir, args.pred_dir, limits=limits)
    print("%d, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f" % (args.iter, np.mean(ate_all), np.std(ate_all), np.mean(te_all), np.std(te_all), np.mean(re_all), np.std(re_all), np.mean(te_all_2), np.std(te_all_2), np.mean(re_all_2), np.std(re_all_2)))

main()
