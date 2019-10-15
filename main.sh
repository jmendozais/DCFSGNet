# PACKAGES TO INSTALL
pip install joblib

# VARIABLES
kitti_raw="/home/juliomb/datasets/kitti/raw_data/"
kitti_odometry="/home/juliomb/datasets/kitti/dataset_odometry_color/"
kitti_sflow_multiview="/home/juliomb/datasets/kitti/data_scene_flow_multiview/"
kitti_sflow="/home/juliomb/datasets/kitti/data_scene_flow/"
kitti_sflow_calib="/home/juliomb/datasets/kitti/data_scene_flow_calib/"

# PREPARE DATASET

# Format depth
#python3 data/prepare_train_data.py --dataset_dir=${kitti_raw} --dataset_name=kitti_raw_eigen --dump_root=kitti_depth_3_rmstatic_new2 --seq_length=3 --img_height=128 --img_width=416 --num_threads=16 --remove_static

# Format flow
#python3 data/prepare_train_data.py --dataset_dir=${kitti_raw} --dataset_name=kitti_raw_stereo --dump_root=kitti_flow_3 --seq_length=3 --img_height=128 --img_width=416 --num_threads=16 

# Format pose
#python3 data/prepare_train_data.py --dataset_dir=${kitti_odometry} --dataset_name=kitti_odom --dump_root=kitti_pose_3_rmstatic --seq_length=3 --img_height=128 --img_width=416 --num_threads=16 --remove_static

# Train rigid fast
#python3 geonet_main.py --mode=train_rigid --dataset_dir=kitti_formated --checkpoint_dir=geonet_rigid_1000 --learning_rate=0.0002 --seq_length=3 --batch_size=4 --max_steps=1000 --tag=rigidfast --save_ckpt_freq=500

# Train rigid

#python3 geonet_main.py --mode=train_rigid --dataset_dir=kitti_depth_3_rmstatic --checkpoint_dir=geonet_depth1 --learning_rate=0.0002 --seq_length=3 --batch_size=4 --max_steps=600000 --tag=depth1

#python3 geonet_main.py --mode=train_rigid --dataset_dir=kitti_pose_5_rmstatic --checkpoint_dir=adv_att_pose_nodo --learning_rate=0.0002 --seq_length=5 --batch_size=4 --max_steps=500000 --tag=adv_att_pose_nodo

#python3 geonet_main.py --mode=train_rigid --dataset_dir=kitti_pose_5_rmstatic --checkpoint_dir=baseline_pose --learning_rate=0.0002 --seq_length=5 --batch_size=4 --max_steps=500000 --tag=baseline_pose

#python3 geonet_main.py --mode=train_rigid --dataset_dir=kitti_pose_5_rmstatic --checkpoint_dir=geonet_rigid_scaled --learning_rate=0.0002 --seq_length=5 --batch_size=4 --max_steps=400000 --tag=rigid_kpose --scale_normalize

# Train flow fast
#python3 geonet_main.py --mode=train_flow --dataset_dir=kitti_formated --checkpoint_dir=geonet_rflow_1000 --learning_rate=0.0002 --seq_length=3 --flownet_type=residual --max_steps=1000 --init_ckpt_file=geonet_rigid_1000/model-500 --tag=rflowfast --save_ckpt_freq=500

# Train flow
#python3 geonet_main.py --mode=train_flow --dataset_dir=kitti_pose_5_rmstatic --checkpoint_dir=geonet_rflow --learning_rate=0.0002 --seq_length=5 --flownet_type=residual --max_steps=400000 --init_ckpt_file=geonet_rigid2/model-350000 --tag=rflow_kpose

# Train flow
#python3 geonet_main.py --mode=train_flow --dataset_dir=kitti_pose_5_rmstatic --checkpoint_dir=geonet_dflow --learning_rate=0.0002 --seq_length=5 --flownet_type=direct --max_steps=400000 --tag=dflow_kpose

# Test depth
#python geonet_main.py --mode=test_depth --dataset_dir=${kitti_raw} --init_ckpt_file=geonet_rigid_last/model-200000 --batch_size=1 --depth_test_split=eigen --output_dir=depth_out

#model_name="rigid_last"
#ckp=200

#model_file="geonet_${model_name}/model-${ckp}000"
#output_dir="${model_name}_depth"
#python geonet_main.py --mode=test_depth --dataset_dir=${kitti_raw} --init_ckpt_file=${model_file} --batch_size=1 --depth_test_split=eigen --output_dir=${output_dir}
#python kitti_eval/eval_depth.py --split=eigen --kitti_dir=${kitti_raw} --pred_file=depth_out/model-200000.npy
#python kitti_eval/eval_depth.py --split=eigen --kitti_dir=${kitti_raw} --pred_file=resnet_eigen.npy


#Test pose
evaluate_pose_old() {
    model_name=$1
    start=$2
    end=$3
    seq=$4
    ckps=()

    for((i=$start; i<=$end; i+=25))
    do
       ckps+=($i)
    done

    for ckp in "${ckps[@]}"
    do
        model_file="geonet_${model_name}/model-${ckp}000"
        pred_dir="${model_name}-pose-${ckp}k/"
        echo "########### $model_file ###########"
        python geonet_main.py --mode=test_pose --dataset_dir=${kitti_odometry} --init_ckpt_file=${model_file} --batch_size=1 --seq_length=3 --pose_test_seq=${seq} --output_dir=${pred_dir}
    done

    for ckp in "${ckps[@]}"
    do
        pred_dir="${model_name}-pose-${ckp}k/"
        python kitti_eval/eval_pose.py --gtruth_dir=pose_gt_${seq}_snpts_s3/ --pred_dir=${pred_dir}
    done
}

evaluate_pose() {
    model_name=$1
    start=$2
    end=$3
    seq=$4
    log=${5:-$end}
    ckps=()

    seq_len=5

    for((i=$start; i<=$end; i+=25))
    do
       ckps+=($i)
    done

    comment='for ckp in "${ckps[@]}"
    do
        model_file="${model_name}/model-${ckp}000"
        pred_dir="${model_name}/seq-${seq}-pose/${ckp}k/"
        echo "########### $model_file ###########"
        if [ "$ckp" == "$log" ]; then
            python geonet_main.py --mode=test_pose --dataset_dir=${kitti_odometry} --init_ckpt_file=${model_file} --batch_size=1 --seq_length=5 --pose_test_seq=${seq} --output_dir=${pred_dir} --log_images
        else
            python geonet_main.py --mode=test_pose --dataset_dir=${kitti_odometry} --init_ckpt_file=${model_file} --batch_size=1 --seq_length=5 --pose_test_seq=${seq} --output_dir=${pred_dir} 
        fi
    done'

    results_dir="${model_name}/seq-${seq}-pose/"
    echo "call start:${start} end:${end} seq:${seq} seq_len:${seq_len}">>${results_dir}pose_log.txt
    for ckp in "${ckps[@]}"
    do
        pred_dir="${model_name}/seq-${seq}-pose/${ckp}k/"
        python kitti_eval/eval_pose.py --gtruth_dir=pose_gt_${seq}_snpts/ --iter=${ckp} --pred_dir=${pred_dir}>>${results_dir}pose_log.txt
    done

}

#evaluate_pose adv_att_pose 25 500 9
#evaluate_pose adv_att_pose_nodo 25 350  9

#model_file="geonet_seq3/model-325000"
#pred_dir="geonet_seq3/pose-325k-s10/"
#pred_dir="seq3-pose-325k/"
#python geonet_main.py --mode=test_pose --dataset_dir=${kitti_odometry} --init_ckpt_file=${model_file} --batch_size=1 --seq_length=3 --pose_test_seq=10 --output_dir=${pred_dir}


#python kitti_eval/generate_pose_snippets.py --dataset_dir=${kitti_odometry} --output_dir=pose_gt_9_snpts_s3/ --seq_id=09 --seq_length=3

#for ckp in "${ckps[@]}"
#do
#	pred_dir="${model_name}-pose-${ckp}k/"
#	echo "########### $pred_dir ###########"
#python kitti_eval/eval_pose.py --gtruth_dir=pose_gt_9_snpts_s3/ --pred_dir=${pred_dir}
#done

#pred_dir="geonet_pose_result_official/09/"
#echo "########### $pred_dir ###########"
#python kitti_eval/eval_pose.py --gtruth_dir=pose_gt_9_snpts/ --pred_dir=${pred_dir}



# Test optical flow
#python kitti_eval/generate_multiview_extension.py --dataset_dir=${kitti_sflow_multiview} --calib_dir=${kitti_sflow_calib} --dump_root=kitti_sflow_5/ --cam_id=02 --seq_length=5
 
#python geonet_main.py --mode=test_flow --dataset_dir=kitti_sflow_5/ --init_ckpt_file=geonet_dflow/model-345000 --flownet_type=direct --batch_size=1 --output_dir=dflow_pred/

#python kitti_eval/eval_flow.py --dataset_dir=${kitti_sflow} --pred_dir=dflow_pred/model-345000/
