import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np


def run():
    # seeds = [0, 1, 2, 3, 4, 5]
    # For color, with 6 training tasks.
    seeds = [10 * i + j for i in range(10) for j in range(3)] + [10 * i + 3 + eval_task for i in range(10)]
    # seeds = [10, 11, 12]
    if domain == 'color':
        spaces = ['rgb']
        global_colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:pink', 'tab:olive', 'tab:cyan']  # List to store colors based on seed % 10
        global_labels = ['Red', 'Green', 'Blue', 'Reward (Red)', 'Reward (Green)', 'Reward (Blue)']
        eval_labels = ['Reward (Red)', 'Reward (Green)', 'Reward (Blue)']
        savedir = 'rgb'
    else:
        spaces = ['grid']
        global_colors = ['r', 'g', 'k']
        global_labels = ['X', 'Y', 'Reward']
        eval_labels = ['Reward']
        savedir = 'grid'
    global_shapes = ['o', '*', 's', 'v', '^', 'x']

    run_type_to_data = {}
    for space in spaces:
        for seed in seeds:
            run_dir = 'saved_data/' + space + '/seed' + str(seed) + '/'
            with open(run_dir + 'distortions' + str(eval_task), 'rb') as file:
                dist = pkl.load(file)
            with open(run_dir + 'comps', 'rb') as file:
                c = pkl.load(file)
            dist = [-d if d is not None else None for d in dist]
            run_type = seed % 10
            safe_c = []
            safe_d = []
            for _c, _d in zip(c, dist):
                if _c is None or _d is None:
                    continue
                safe_c.append(_c)
                safe_d.append(_d)

            if run_type_to_data.get(run_type) is None:
                run_type_to_data[run_type] = [[], []]
            run_type_to_data.get(run_type)[0].extend(safe_c)
            run_type_to_data.get(run_type)[1].extend(safe_d)

    for run_type, data in run_type_to_data.items():
        c, dist = data
        label = 'Training dimension: ' + global_labels[run_type]
        plt.scatter(c, dist, c=global_colors[run_type], marker=global_shapes[run_type], label=label)

        # Polynomial fit
        poly_fit_degree = 3  # You can change the degree as needed
        coeffs = np.polyfit(c, dist, poly_fit_degree)
        poly = np.poly1d(coeffs)
        c_range = np.linspace(min(c), max(c), 100)
        plt.plot(c_range, poly(c_range), color=global_colors[run_type], linestyle='dashed')

    plt.xlabel("Complexity (bits)")
    plt.ylabel("Negative " + eval_labels[eval_task] + " Distortion")
    plt.title("Task Distortion vs. Complexity")
    plt.legend()
    plt.savefig('saved_data/' + savedir + '/all_runs' + str(eval_task) + '.png')
    plt.close()


if __name__ == '__main__':
    ib_model = None
    # domain = 'grid'
    domain = 'color'
    eval_task = 0
    run()
