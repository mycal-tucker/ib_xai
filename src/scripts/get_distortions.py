
import os
import pickle as pkl

import numpy as np
import torch
from ib_color_naming.src import ib_naming_model

import src.settings as settings
from src.utils.helper_fns import train_task_head, get_color_data, get_grid_data


def run():
    # eval_tasks = [0, 1, 2]
    eval_tasks = [3, 4, 5]
    # seeds = [0, 1, 2, 3, 4, 5]
    seeds = [i + 10 * j for i in range(6) for j in range(10)]
    for eval_task in eval_tasks:
        if domain == 'color':
            _, data = get_color_data(ib_model, batch_size, eval_dim=eval_task)
        else:
            _, data = get_grid_data(batch_size, train_dim=0)  # Train dim doesn't matter
        for seed in seeds:
            print("\n***************Evaluating seed", seed, "for eval task", eval_task, "******************")
            distortions = []
            # Iterate over all checkpoints in that saved seed
            if domain == 'color':
                seed_dir = 'saved_data/rgb/seed' + str(seed)
            else:
                seed_dir = 'saved_data/grid/seed' + str(seed)
            check_vals = []
            for checkpoint in os.listdir(seed_dir):
                checkpoint_dir = seed_dir + '/' + checkpoint
                if not os.path.isdir(checkpoint_dir) or checkpoint == 'pca':
                    continue
                if int(checkpoint) % 500 != 0:
                    check_vals.append(int(checkpoint))
                    distortions.append(None)
                    continue  # Subsample to only look at some checkpoints. Speeds things up a lot.
                print("Checkpoint", checkpoint)
                ae = torch.load(checkpoint_dir + '/ae.pt')
                dist = train_task_head(ae, data)
                distortions.append(dist)
                check_vals.append(int(checkpoint))

            # Sort distortions based on check_vals
            distortions = [dist for _, dist in sorted(zip(check_vals, distortions))]
            with open(seed_dir + '/distortions' + str(eval_task), 'wb') as file:
                print("Saving distortions to", seed_dir + '/distortions' + str(eval_task))
                pkl.dump(distortions, file)


if __name__ == '__main__':
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # domain = 'grid'
    domain = 'color'
    if domain == 'color':
        settings.color_space = 'rgb'
        batch_size = 1024
        ib_model = ib_naming_model.load_model()
        prior = ib_model.pM[:, 0]
        prior = np.ones_like(prior) / 330
        ib_model.pM[:, 0] = prior
    else:
        batch_size = 1024
    run()
