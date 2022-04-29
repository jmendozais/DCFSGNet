# PACKAGES TO INSTALL
pip install joblib

# VARIABLES
#kitti_raw="/home/juliomb/datasets/kitti/raw_data/"
kitti_raw="/data/KITTI/raw_data/"
kitti_odometry="/home/juliomb/datasets/kitti/dataset_odometry_color/"
kitti_sflow_multiview="/home/juliomb/datasets/kitti/data_scene_flow_multiview/"
kitti_sflow="/home/juliomb/datasets/kitti/data_scene_flow/"
kitti_sflow_calib="/home/juliomb/datasets/kitti/data_scene_flow_calib/"

# PREPARE DATASET

# Format depth
#python3 data/prepare_train_data.py --dataset_dir=${kitti_raw} --dataset_name=kitti_raw_eigen --dump_root=kitti_depth_3_rmstatic --seq_length=3 --img_height=128 --img_width=416 --num_threads=16 --remove_static

# Format pose
#python3 data/prepare_train_data.py --dataset_dir=${kitti_odometry} --dataset_name=kitti_odom --dump_root=kitti_pose_3_rmstatic --seq_length=3 --img_height=128 --img_width=416 --num_threads=16 --remove_static

# Train rigid

#python3 geonet_main.py --mode=train_rigid --dataset_dir=kitti_depth_3_rmstatic --checkpoint_dir=advatt_ap_de --learning_rate=0.0002 --seq_length=3 --batch_size=4 --max_steps=500000 --tag=advatt_ap_de

#python3 geonet_main.py --mode=train_rigid --dataset_dir=kitti_depth_3_rmstatic --checkpoint_dir=baseline_nodo_44k_nodo --learning_rate=0.0002 --seq_length=3 --batch_size=4 --max_steps=500000 --tag=baseline_nodo_44k_nodo

#python3 geonet_main.py --mode=train_rigid --dataset_dir=kitti_depth_3_rmstatic --checkpoint_dir=baseline_nodo_44k --learning_rate=0.0002 --seq_length=3 --batch_size=4 --max_steps=500000 --tag=baseline_nodo_44k

#python3 geonet_main.py --mode=train_rigid --dataset_dir=kitti_depth_3_rmstatic --checkpoint_dir=adv_att_depth --learning_rate=0.0002 --seq_length=3 --batch_size=4 --max_steps=500000 --tag=adv_att_depth

#python3 geonet_main.py --mode=train_rigid --dataset_dir=kitti_depth_3_rmstatic --checkpoint_dir=geonet_depth1 --learning_rate=0.0002 --seq_length=3 --batch_size=4 --max_steps=600000 --tag=depth1

#python3 geonet_main.py --mode=train_rigid --dataset_dir=kitti_pose_3_rmstatic --checkpoint_dir=geonet_seq3_long --learning_rate=0.0002 --seq_length=3 --batch_size=4 --max_steps=1000000 --tag=seq3_long

#python3 geonet_main.py --mode=train_rigid --dataset_dir=kitti_pose_5_rmstatic --checkpoint_dir=geonet_rigid_scaled --learning_rate=0.0002 --seq_length=5 --batch_size=4 --max_steps=400000 --tag=rigid_kpose --scale_normalize

# Test depth
#python geonet_main.py --mode=test_depth --dataset_dir=${kitti_raw} --init_ckpt_file=geonet_rigid_last/model-200000 --batch_size=1 --depth_test_split=eigen --output_dir=depth_out

#model_name="rigid_last"
#ckp=200

#model_file="geonet_${model_name}/model-${ckp}000"
#output_dir="${model_name}_depth"
#python geonet_main.py --mode=test_depth --dataset_dir=${kitti_raw} --init_ckpt_file=${model_file} --batch_size=1 --depth_test_split=eigen --output_dir=${output_dir}

#python kitti_eval/eval_depth.py --split=eigen --kitti_dir=${kitti_raw} --pred_file=depth_out/model-200000.npy
#python kitti_eval/eval_depth.py --split=eigen --kitti_dir=${kitti_raw} --pred_file=resnet_eigen.npy

evaluate_depth() {
    model_name=$1
    start=$2
    end=$3
    step=$4
    log=${5:-$end}
    ext_args=${6:-''}
    ckps=()

    for((i=$start; i<=$end; i+=$step))
    do
       ckps+=($i)
    done

    for ckp in "${ckps[@]}"
    do
        #model_file="${model_name}/model-${ckp}000"
        #pred_dir="${model_name}/depth/${ckp}k/"
        model_file="/data/ra153646/tflogs/${model_name}/model-${ckp}000"
        pred_dir="/data/ra153646/tflogs/${model_name}/depth/${ckp}k/"
        echo "########### $model_file - depth ###########"
        echo python geonet_main.py --mode=test_depth --dataset_dir=${kitti_raw} --init_ckpt_file=${model_file} --batch_size=1 --depth_test_split=eigen --output_dir=${pred_dir} ${ext_args}
        python geonet_main.py --mode=test_depth --dataset_dir=${kitti_raw} --init_ckpt_file=${model_file} --batch_size=1 --depth_test_split=eigen --output_dir=${pred_dir} ${ext_args}
    done

    #results_dir="${model_name}/depth/"
    results_dir="/data/ra153646/tflogs/${model_name}/depth/"
    for ckp in "${ckps[@]}"
    do
        pred_dir="${results_dir}${ckp}k/"
        if [ "$ckp" == "$log" ]; then
            python kitti_eval/eval_depth.py --split=eigen --kitti_dir=${kitti_raw} --iter=${ckp} --log --pred_file=${pred_dir}depth.npy>>${results_dir}depth_log.txt
	else
            python kitti_eval/eval_depth.py --split=eigen --kitti_dir=${kitti_raw} --iter=${ckp} --pred_file=${pred_dir}depth.npy>>${results_dir}depth_log.txt 
	fi
    done
}

# model_name="depth_dc_att0.5l2_th0.3_w0.02_socc0.5_cons_socc_bi"
# model_name="depth_dc_att0.5l2_th0.3_w0.02_socc0.5_cons_socc_do"
# model_name="depth_dc_att0.5l2_th0.3_w0.02_socc0.5_cons_socc"
# model_name="dc_att0.7l2regnobn_th0.3_w0.2_conssocc0.5"
#evaluate_depth $model_name 25 600

#evaluate_depth "baseline" 25 275 275
#evaluate_depth "baseline_nodo" 25 500 275
#evaluate_depth "baseline_nodo_44k" 5 500 5 500
#evaluate_depth "baseline_nodo_44k_nodo" 5 500 5 500

#evaluate_depth "adv_att_depth" 25 300 275
#evaluate_depth "adv_att_depth" 300 500
#evaluate_depth "adv_att_depth" 150 150 150

#evaluate_depth "advatt_depth_w1" 5 500 5 500 
#evaluate_depth "advatt_depth_w0.31" 5 500 5 500 
#evaluate_depth "advatt_depth_w3.1" 5 350 5 350 
#evaluate_depth "advatt_depth_w0.1" 5 500 5 500 
#evaluate_depth "advatt_depth_w10" 5 290 5 290
#evaluate_depth "advatt_depth_w10" 295 465 5 465

#evaluate_depth "advatt_ap_de_w1" 5 220 5 220 
#evaluate_depth "advatt_ap_de_hd_w1" 5 320 5 320 
#evaluate_depth "advatt_ap_de_hd_w1" 325 465 5 465 #<-TODO
#evaluate_depth "advatt_ap_de_w3.1_a0.9" 5 420 5 420
#evaluate_depth "advatt_ap_de_w3.1_a0.99" 280 360 5 360
#evaluate_depth "advatt_ap_de_w1_a0.99" 5 320 5 320
#evaluate_depth "advatt_ap_de_dc0.1_w3.1_a0.9_train" 5 500 5 500

#evaluate_depth advatt_ap_de_w1_a0.99_nossim 5 210 5 210

#evaluate_depth "baseline_rd_0.033" 280 500 10 500
#evaluate_depth "baseline_rd_0.01" 330 500 10 500
#evaluate_depth "baseline_rd_0.1" 10 500 10 500
#evaluate_depth "baseline_rd_0.1" 390 390 10 390
#evaluate_depth "baseline_rd_0.31" 50 500 10 500 
#evaluate_depth "baseline_rd_1" 10 500 10 500 
#evaluate_depth "baseline_rd_0.31" 430 430 10 430 
#evaluate_depth "baseline_rd_0.31_train" 10 500 10 500 

#evaluate_depth "baseline_rd_0.0033" 10 500 10 500
#evaluate_depth "baseline_rd_0.31_train" 10 360 10 360
#evaluate_depth "coupled_dc0.31_train_nosg" 5 210 5 210
#evaluate_depth "coupled_dc0.31_train_nosg" 5 390 5 390
#evaluate_depth "coupled_dc0.31_train" 10 450 10 450
#evaluate_depth "coupled_dc0.31_train_complete" 5 300 5 300
#evaluate_depth "coupled_dc0.31_lr0.0001_train_complete" 5 320 5 320
#evaluate_depth "co_dc0.31_a2l_k3_train" 200 480 5 480
#evaluate_depth "fcsimp_dc0.31_train" 5 180 5 180
#evaluate_depth "fcsimp_dc0.31_train" 305 305 5 305
#evaluate_depth "fcsimp_dc_nodo_0.31__train" 5 500 5 500
#evaluate_depth "fcsimp_dc_do1_0.31__train" 5 500 5 500
evaluate_depth "fcsimp_dc_do3_0.31__train" 5 500 5 500

#model_file="geonet_seq3/model-325000"
#pred_dir="geonet_seq3/pose-325k-s10/"
#pred_dir="seq3-pose-325k/"
#python geonet_main.py --mode=test_pose --dataset_dir=${kitti_odometry} --init_ckpt_file=${model_file} --batch_size=1 --seq_length=3 --pose_test_seq=10 --output_dir=${pred_dir}

#python kitti_eval/eval_pose.py --gtruth_dir=pose_gt_9_snpts_s3/ --pred_dir=${pred_dir}

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

