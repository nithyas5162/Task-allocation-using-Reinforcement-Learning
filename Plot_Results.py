import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plot_Results():
    for a in range(1):
        Eval = np.load('Evaluate_all.npy', allow_pickle=True)[a]

        Terms = ['Cost', 'Throughput', 'Delay', 'Time',
                 'Load', 'Task Waiting Time', 'Task Completion Time', 'CPU Utilization', 'Memory Utilization']
        for b in range(len(Terms)):
            No_of_Task = [50, 100, 150, 200, 250]

            X = np.arange(5)
            plt.plot(No_of_Task, Eval[:, 0, b], color='g', linewidth=3, marker='o', markerfacecolor='g', markersize=14,
                     label="RPO")
            plt.plot(No_of_Task, Eval[:, 1, b], color='#ad03de', linewidth=3, marker='o', markerfacecolor='#ad03de',
                     markersize=14,
                     label="DSOA")
            plt.plot(No_of_Task, Eval[:, 2, b], color='#8c564b', linewidth=3, marker='o', markerfacecolor='#8c564b',
                     markersize=14,
                     label="WSA")
            plt.plot(No_of_Task, Eval[:, 3, b], color='b', linewidth=3, marker='o', markerfacecolor='b', markersize=14,
                     label="BIO")
            plt.plot(No_of_Task, Eval[:, 4, b], color='k', linewidth=3, marker='o', markerfacecolor='k', markersize=14,
                     label="HWSBIA")

            labels = ['50', '100', '150', '200', '250']
            plt.xticks(No_of_Task, labels)

            plt.xlabel('No of Task')
            plt.ylabel(Terms[b])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/%s_line.png" % (Terms[b])
            plt.savefig(path1)
            plt.show()


def plot_Fitness():
    No_of_Task = [50, 100, 150, 200, 250]
    for a in range(5):  # For 5 Configurations
        Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
        Algorithm = ['RPO', 'DSOA', 'WSA', 'BIO', 'HWSBIA']

        conv = np.load('Fitness.npy', allow_pickle=True)[a]
        ind = np.argsort(conv[:, conv.shape[1] - 1])
        x = conv[ind[0], :].copy()
        y = conv[4, :].copy()
        conv[4, :] = x
        conv[ind[0], :] = y

        Value = np.zeros((conv.shape[0], 5))
        for j in range(conv.shape[0]):
            Value[j, 0] = np.min(conv[j, :])
            Value[j, 1] = np.max(conv[j, :])
            Value[j, 2] = np.mean(conv[j, :])
            Value[j, 3] = np.median(conv[j, :])
            Value[j, 4] = np.std(conv[j, :])

        Table = PrettyTable()
        Table.add_column("ALGORITHMS", Statistics)
        for j in range(len(Algorithm)):
            Table.add_column(Algorithm[j], Value[j, :])
        print('-------------------------------------------------- Task-' + str(
            No_of_Task[a]) + '-Statistical Analysis--------------------------------------------------')
        print(Table)

        iteration = np.arange(conv.shape[1])
        plt.plot(iteration, conv[0, :], color='#aaff32', linewidth=5,
                 label="RPO")
        plt.plot(iteration, conv[1, :], color='b', linewidth=5,
                 label="DSOA")
        plt.plot(iteration, conv[2, :], color='y', linewidth=5,
                 label="WSA")
        plt.plot(iteration, conv[3, :], color='#ff0490', linewidth=5,
                 label="BIO")
        plt.plot(iteration, conv[4, :], color='k', linewidth=5,
                 label="HWSBIA")
        plt.xlabel('No. of Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        path1 = "./Results/Task-%s.jpg" % (str(a + 1))
        plt.savefig(path1)
        plt.show()


def plot_Reward():
    Vary_No_of_Statesize = [12, 24, 36, 48, 60]
    Algorithms = ['RPO', 'DSOA', 'WSA', 'BIO', 'HWSBIA']
    Maximum_Rewards = np.load('Reward.npy', allow_pickle=True)

    for n in range(len(Vary_No_of_Statesize)):
        Rewards = Maximum_Rewards[n]

        X = np.arange(Rewards.shape[1])
        plt.plot(X, Rewards[0, :], color='c', linewidth=3, marker='*', markerfacecolor='r', markersize=16,
                 label="RPO")
        plt.plot(X, Rewards[1, :], color='g', linewidth=3, marker='*', markerfacecolor='g', markersize=16,
                 label="DSOA")
        plt.plot(X, Rewards[2, :], color='b', linewidth=3, marker='*', markerfacecolor='b', markersize=16,
                 label="WSA")
        plt.plot(X, Rewards[3, :], color='m', linewidth=3, marker='*', markerfacecolor='m', markersize=16,
                 label="BIO")
        plt.plot(X, Rewards[4, :], color='k', linewidth=3, marker='*', markerfacecolor='k', markersize=16,
                 label="HWSBIA")
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        # plt.ylim([80, 100])
        plt.legend(loc=4)
        path1 = "./Results/reward_%s.png" % (n + 1)
        plt.savefig(path1)
        plt.show()


if __name__ == '__main__':
    plot_Results()
    plot_Fitness()
    plot_Reward()
