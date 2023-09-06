import numpy as np 
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='WikiCS')
args = parser.parse_args()

a = np.loadtxt(f'results/{args.dataset}.txt')

f = open(f'results/{args.dataset}.txt','a')
print(f'{a.mean()}, {a.std()}', file=f)
f.close()
