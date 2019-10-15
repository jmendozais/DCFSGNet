# VARIABLES
kitti_raw="/home/juliomb/datasets/kitti/raw_data_downloader/"
kitti_odometry="/home/juliomb/datasets/kitti/dataset_odometry_color/"
kitti_sflow_multiview="/home/juliomb/datasets/kitti/data_scene_flow_multiview/"
kitti_sflow="/home/juliomb/datasets/kitti/data_scene_flow/"
kitti_sflow_calib="/home/juliomb/datasets/kitti/data_scene_flow_calib/"

model_dir=$1

tmp_file=$(tempfile)
out_file=${1}/ate_pred.txt
echo $out_file
echo $tmp_file
for f in ${model_dir}/*meta; do
    model_ckpt=${f%.meta}
    echo $f
    python geonet_main.py --mode=test_pose --dataset_dir=${kitti_odometry} --init_ckpt_file=$model_ckpt --batch_size=1 --seq_length=5 --pose_test_seq=9 --output_dir=${model_ckpt}_pred/
    python kitti_eval/eval_pose.py --gtruth_dir=pose_gt_9_snpts/ --pred_dir=${model_ckpt}_pred/ >>${tmp_file}
done
cat ${tmp_file} | grep 'ATE' >${out_file}

