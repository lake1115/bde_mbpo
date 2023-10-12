import argparse
import re
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def get_dir():
    root_dir = "exp/bde_mbpo/"
    experiment_dir = os.listdir(root_dir)

    # get all the experiment dirs
    experiment_dir = [os.path.join(root_dir, exp_dir) for exp_dir in experiment_dir]

    # get all the experiment names
    experiment_names = [os.listdir(exp_dir) for exp_dir in experiment_dir]

    for i in range(len(experiment_names)):
        experiment_names[i] = [os.path.join(experiment_dir[i], exp_name) for exp_name in experiment_names[i]]

    return experiment_names[0]

def plot_with_filterd(filtered_name
                      , pattern = r'rl\d+_Ns\d+'
                      , filename='results.csv'
                      , y_label='episode_reward'
                      , save_dir=None):
    '''
    Args:
        - filtered_name
        - pattern: the pattern to be matched
        - filename: the file name to be plotted
        - save_dir: None or the directory to save the figure
    '''
    data_model_lst = []
    labels = []
    for i, name in enumerate(filtered_name):
        df = pd.read_csv(os.path.join(name, filename))
        
        # indicx = df.columns
        # indicx = list(filter(lambda x: ('epoch' == x) or ('iteration' == x) or ('step' == x), indicx))
        # if indicx == []:
        #    data_model_lst.append(df)
        #else:
        data_model_lst.append(df.sort_values(by=['step']))
        label = filtered_name[i].split("/")[-1]
        labels.append(re.findall(pattern, label))

    # plot the results
    plt.figure(figsize=(10, 4), dpi=300)
    for i, data in enumerate(data_model_lst):
        plt.plot(data.iloc[0:, :][y_label], label=labels[i])
        
    plt.legend()
    plt.xlabel("steps")
    plt.ylabel(y_label)
    plt.title(f"{y_label} vs. Steps")
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{y_label}_vs_steps.png"))
    else:
        plt.show()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Args for Visualization")
    parser.add_argument("--filter_condition", type=str, default="after_debug")
    parser.add_argument("--save_dir", type=str, default="./")
    args = parser.parse_args()

    experiment_names = get_dir()
   
    filtered_name = experiment_names
    plot_with_filterd(filtered_name, pattern = r'rl\d+_Ns\d+', filename='results.csv', y_label='episode_reward', save_dir=args.save_dir)
    plot_with_filterd(filtered_name
                      , pattern = r'rl\d+_Ns\d+', filename='model_train.csv'
                      , y_label='model_loss', save_dir=args.save_dir)
    plot_with_filterd(filtered_name, pattern = r'rl\d+_Ns\d+'
                      , filename='train.csv', save_dir=args.save_dir
                      , y_label='actor_loss')
