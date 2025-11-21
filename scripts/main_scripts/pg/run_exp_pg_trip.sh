EXPNAME="Main"
EXPNAME="P4G_TRIP_PG"
for i in 1
do
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 8080 --gpu_ids 1 --num_processes 1 run.py  \
        --exp_name $EXPNAME \
        --project_name ProactiveLLM \
        --seed $i \
        --scenario persuation \
        --log_dir logs \
        --loggers terminal,file,wandb \
        --datasets p4g \
        --models trip \
        --is_so_game \
        --num_train_rl_epochs 10 \
        --gen_models chatgpt \
        --model_type chatgpt \
        --metrics acc,prf1,sr,total_reward,avg_turn
done
