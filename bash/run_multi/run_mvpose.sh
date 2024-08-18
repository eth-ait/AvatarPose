seqs=('pair14_talk14')

for seq in ${seqs[@]}; do
    DATA_ROOT=./Hi4D_AvatarPose
    data=$DATA_ROOT/$seq
    python3 third_parties/Easymocap/apps/demo/smpl_from_keypoints.py ${data} --skel ${data}/skel19_4DA --out ${data}/smpl_init --verbose  --body skel19
done