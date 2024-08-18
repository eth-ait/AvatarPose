export PYTHONPATH=$(pwd)
experiment="exp"
avatar_pose_name='est_kpts_opt_1'
SEQUENCES=("hi4d_14_talk14")
for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="hi4d/$SEQUENCE"
    python validate_avatar_smpl.py --config-name opt_avatar_smpl dataset=$dataset experiment=$experiment model.opt.vis_log_name=$avatar_pose_name
    python nv_avatar.py --config-name opt_avatar_smpl dataset=$dataset experiment=$experiment model.opt.vis_log_name=$avatar_pose_name
    python vis_smpl_kpts.py --config-name opt_avatar_smpl dataset=$dataset experiment=$experiment model.opt.vis_log_name=$avatar_pose_name
done