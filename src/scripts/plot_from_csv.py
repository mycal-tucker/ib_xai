import matplotlib.pyplot as plt
import matplotlib
import pandas as pd


def run():
    markers = ['o', 's', 'v']
    # colors = ['r', 'c', 'b']
    for eval_rew in eval_rews:
        plt.figure(figsize=(6, 4))
        for i, train_rew in enumerate(train_rews):
            ib_data = pd.read_csv('./saved_data/' + domain + '/train_' + train_rew + '/ib_results.csv')
            comp = ib_data['Complexity']
            dist = ib_data[eval_rew]
            plt.scatter(comp, dist, c=colors[i], label=labels[i], marker=markers[i], s=80)
            plt.plot(comp, dist, c=colors[i])
        plt.legend(loc='upper right')
        if domain == 'exact_ib_rgb':
            plt.title('Cont. Blue IB Tradeoff')
        else:
            plt.title('Manhattan Grid IB Tradeoff')
        plt.xlabel("Complexity (bits)")
        plt.ylabel("Distortion (MSE)")
        plt.tight_layout()
        plt.savefig('./saved_data/' + domain + '/' + eval_rew + '_IB')
        plt.close()


if __name__ == '__main__':
    font = {'family': 'normal',
            'size': 18}
    matplotlib.rc('font', **font)

    # domain = 'exact_ib_rgb'
    # eval_rews = ['blue_cont']
    # train_rews = ['red_cont', 'blue_disco',  'blue_cont']
    # labels = ['Red Cont.', 'Blue Disc.', 'Blue Cont.', ]
    # colors = ['r', 'c', 'b']

    domain = 'exact_ib_grid/grid4'
    eval_rews = ['Reward Distortion']
    train_rews = ['x_val', 'rew_val']
    labels = ['$X$', 'Reward']
    colors = ['r', 'b']
    run()
