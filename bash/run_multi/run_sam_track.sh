seqs=('pair14_talk14')

for seq in ${seqs[@]}; do
    DATA_ROOT=./Hi4D_AvatarPose
    cd third_parties/sam_track
    python sta.py --data_root $DATA_ROOT --seq $seq
done