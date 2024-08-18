seqs=('pair14/talk14')

for seq in ${seqs[@]}; do
    CAPTURE_ROOT=./Hi4D
    DATA_ROOT=./Hi4D_AvatarPose
    python scripts/preprocess/pre_hi4d.py --indir $CAPTURE_ROOT --outdir $DATA_ROOT --seq $seq
done