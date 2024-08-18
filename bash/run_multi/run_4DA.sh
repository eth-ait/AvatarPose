seqs=('pair14_talk14')

for seq in ${seqs[@]}; do
    DATA_ROOT=./Hi4D_AvatarPose
    OPENPOSE_ROOT=./openpose/openpose
    python scripts/fDAssociation/run_4D.py --data_root $DATA_ROOT --seq $seq --openpose $OPENPOSE_ROOT
done