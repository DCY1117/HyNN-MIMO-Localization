# for regression
datasets_reg=(
    DIS_lab_LoS_16_X
    DIS_lab_LoS_16_Y
    URA_lab_LoS_8_X
    URA_lab_LoS_8_Y
    URA_lab_LoS_16_X
    URA_lab_LoS_16_Y
    ULA_lab_LoS_8_X
    ULA_lab_LoS_8_Y
    ULA_lab_LoS_16_X
    ULA_lab_LoS_16_Y
)
models=(
    xgboost
)
indices_models=(
)

tabr_ohe_models=(
)
for dataset in "${datasets_reg[@]}"; do
    for model in "${models[@]}"; do
        python ./train_model_classical.py --dataset $dataset --dataset_path data --tune --n_trial 50 --seed_num 1 --model_type $model --gpu 0 > "./log/${dataset}-${model}.txt"
    done
    for model in "${indices_models[@]}"; do
        python ./train_model_classical.py --dataset $dataset --dataset_path data --tune --n_trial 20 --seed_num 1  --model_type $model --gpu 0 --cat_policy indices > "./log/${dataset}-${model}.txt"
    done
    for model in "${tabr_ohe_models[@]}"; do
        python ./train_model_classical.py --dataset $dataset --dataset_path data --tune --n_trial 20 --seed_num 1  --model_type $model --gpu 0 --cat_policy tabr_ohe  > "./log/${dataset}-${model}.txt"
    done
done