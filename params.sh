# Parameter evaluation

run() {
task=$1
seq_length=3
dataset_dir='kitti_depth_3_rmstatic'
if [ ${task} == 'pose' ]; then
   seq_length=5
   dataset_dir='kitti_pose_5_rmstatic'
fi
consistency_weight=0.31
name=fcsimp_dc_do1_${consistency_weight}_${task}_train
python3 geonet_main.py --mode=train_rigid --dataset_dir=${dataset_dir} --checkpoint_dir=/data/ra153646/tflogs/${name} --seq_length=${seq_length} --batch_size=4 --max_steps=500000 --tag=/data/ra153646/tflogs/${name} --consistency_weight=${consistency_weight}
}

#run 'pose'
run 


