import argparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Args for Visualization")
    parser.add_argument("--data_dir", type=str, default="./exp/bde_mbpo/bde_long_epoch_length/cartpole_continuous_rl3_2023.09.27_200531/model_weights.csv")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--iteration", type=int, default=1)
    args = parser.parse_args()

    df_model_weights = pd.read_csv(args.data_dir)
    df_model_weights = df_model_weights.drop(columns=['Unnamed: 0'])
    df_model_weights = df_model_weights[df_model_weights['epoch'] != 'epoch']
    df_model_weights = df_model_weights.astype({'epoch': 'int32', 'iteration': 'int32'})
    df_model_weights['iteration'] = df_model_weights['iteration'].apply(lambda x: (x + 1)/250-1)

    # plot the results
    condition = (df_model_weights['epoch'] == args.k) & (df_model_weights['iteration'] == args.iteration)
    df = df_model_weights[condition]
    df = df.drop(columns=['epoch', 'iteration', 'index'])
    plt.figure(figsize=(10, 4), dpi=300)
    sns.kdeplot(data=df)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    plt.savefig(os.path.join(args.save_dir,f'iter{args.iteration}_k_{args.k}.png'))

    # python vis_model_weights.py --data_dir ./exp/bde_mbpo/bde_long_epoch_length/cartpole_continuous_rl3_2023.09.27_200531/model_weights.csv --save_dir ./model_weights_test --k 1 --iteration 20