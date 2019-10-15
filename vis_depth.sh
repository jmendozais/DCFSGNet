# PACKAGES TO INSTALL
pip install joblib
pip install pypcd

# VARIABLES
kitti_raw="/home/juliomb/datasets/kitti/raw_data_downloader/"
kitti_odometry="/home/juliomb/datasets/kitti/dataset_odometry_color/"
kitti_sflow_multiview="/home/juliomb/datasets/kitti/data_scene_flow_multiview/"
kitti_sflow="/home/juliomb/datasets/kitti/data_scene_flow/"
kitti_sflow_calib="/home/juliomb/datasets/kitti/data_scene_flow_calib/"

python3 depth2pcd.py --depth_file=depth_out/model-385000.npy --out_file=depth_out/model-385000.pcd

cp depth_out/*.pcd /home/juliomb/public_html/slam/
cp depth_out/*.png /home/juliomb/public_html/slam/
