rm results/$1.txt

python train.py --dataset $1  --seed 1
python train.py --dataset $1  --seed 2
python train.py --dataset $1  --seed 3
python train.py --dataset $1  --seed 4
python train.py --dataset $1  --seed 5

python cal_results.py --dataset $1