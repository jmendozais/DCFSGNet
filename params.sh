# Parameter evaluation
run_val_mode() {
    echo "TODO"
}

run_train_mode() {
    python3 geonet_main.py --mode=train_rigid --dataset_dir=kitti_depth_3_rmstatic --checkpoint_dir=co_dc0.31_a2l_k3_train --seq_length=3 --batch_size=4 --max_steps=500000 --tag=co_dc0.31_a2l_k3_train --consistency_weight=0.31
}

run_train_mode
#TODO: run_train_mode 0.00002


