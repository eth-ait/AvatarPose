seqs=('pair14_talk14')

for seq in ${seqs[@]}; do
    DATA_ROOT=./Hi4D_AvatarPose
    python scripts/mvpose/convert_smpl.py --data_root $DATA_ROOT --seq $seq
done