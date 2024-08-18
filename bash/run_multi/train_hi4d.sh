export PYTHONPATH=$(pwd)
experiment="exp"
avatar_pose_name='est_kpts_opt_1'
SEQUENCES=("hi4d_14_talk14")
for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="hi4d/$SEQUENCE"
    python train_avatar_smpl.py --config-name opt_avatar_smpl dataset=$dataset experiment=$experiment model.opt.vis_log_name=$avatar_pose_name dataset.split_opt.val.ranges=[71,72,1] \
    +model.opt.stages.avatar_betas=[[0,10]] model.opt.stages.avatar=[[40,50],[90,100],[140,150]] \
    +model.opt.stages.arefine_betas=[[10,30]] model.opt.stages.arefine=[[50,70],[100,120],[150,160]] \
    +model.opt.stages.pose_transl_rot=[[30,40],[70,90]] model.opt.stages.pose=[[120,140],[160,180]] \
    train.max_epochs=180 train.check_val_every_n_epoch=10
    python save_smpl.py --config-name opt_avatar_smpl dataset=$dataset experiment=$experiment model.opt.vis_log_name=$avatar_pose_name
    python validate_avatar_smpl.py --config-name opt_avatar_smpl dataset=$dataset experiment=$experiment model.opt.vis_log_name=$avatar_pose_name
done