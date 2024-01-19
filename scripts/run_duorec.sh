gpu_id=$1
for dataset in amazon-beauty amazon-sports-outdoors amazon-toys-games
do
  python run_seq.py \
    --dataset="${dataset}" \
    --config_files "seq.yaml config/config_d_${dataset}.yaml config/config_t.yaml config/gpu_${gpu_id}.yaml" \
    --train_batch_size=256 \
    --lmd=0.1 \
    --lmd_sem=0.1 \
    --model='DuoRec'\
    --contrast='us_x' \
    --sim='dot' \
    --tau=1
done