import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def visualize_3(data):
    g = sns.regplot(data=data, x='avg_energy', y='training_time_s')
    plt.show()


def visualize_baseline(data):
    g = sns.PairGrid(data=data, y_vars=['model_name'],
                     x_vars=['training_time', 'accuracy',
                             'memo_usage_mib', 'cpu_percent_util_max'],
                     hue='model_name', palette="vlag")
    g.map(sns.boxplot)
    g.add_legend()
    g.savefig('baseline-model-metrics.png')

    plt.clf()

    m = sns.PairGrid(data=data, y_vars=['model_name'],
                     x_vars=['total_energy_skylake', 'total_energy_broadwell', 'total_energy_haswell'],
                     hue='model_name', palette='vlag')
    m.map(sns.boxplot)
    m.set(xscale='log')
    m.add_legend()
    m.savefig('baseline_energy.png')

    plt.clf()

    n = sns.PairGrid(data=data, y_vars=['model_name'], x_vars=['power_use_memory',
                                                               'power_use_skylake_cpu',
                                                               'power_use_broadwell_cpu',
                                                               'power_use_haswell_cpu'],
                     hue='model_name', palette='vlag')
    n.map(sns.boxplot)
    m.set(xscale='log')
    n.add_legend()
    n.savefig('baseline_power.png')


def visualise_2(data):
    g = sns.FacetGrid(data=data, col='num_layers', hue='model_name', palette="vlag", xlim=(0, 30))
    g.map(sns.regplot, "max_epoch", "training_time_s")
    g.add_legend()
    g.savefig('experiment-time-epoch-layers.png')

    g = sns.FacetGrid(data=data, col='num_layers', hue='model_name', palette="vlag", xlim=(0, 30))
    g.map(sns.regplot, "max_epoch", "avg_energy")
    g.set(yticks=[10000, 100000, 1000000], yscale="log")
    g.add_legend()
    g.savefig('experiment-energy-epoch-layers.png')

    g = sns.FacetGrid(data=data, col='batch_size', hue='model_name', palette="vlag", xlim=(0, 30))
    g.map(sns.regplot, "max_epoch", "training_time_s")
    g.add_legend()
    g.savefig('experiment-time-epoch-batch.png')

    g = sns.FacetGrid(data=data, col='batch_size', hue='model_name', palette="vlag", xlim=(0, 30))
    g.set(yticks=[10000, 100000, 1000000], yscale="log")
    g.map(sns.regplot, "max_epoch", "avg_energy")
    g.add_legend()
    g.savefig('experiment-energy-epoch-batch.png')

    print('==========DONE==========')


def visualise(data):
    # normalise
    # data.iloc[:, 1:] = data.iloc[:, 1:].apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    # print(data)

    g = sns.PairGrid(data=data, y_vars=['model_name'],
                     x_vars=['training_time_s', 'accuracy',
                             'memo_usage_mib', 'cpu_percent_util_max'],
                     hue='model_name', palette="vlag")
    g.map(sns.boxplot)
    g.add_legend()
    g.savefig('experiment-model-metrics.png')

    plt.clf()

    g = sns.PairGrid(data=data, y_vars=['max_epoch'],
                     x_vars=['training_time_s', 'accuracy',
                             'memo_usage_mib', 'cpu_percent_util_max'],
                     hue='model_name', palette="vlag")
    # g.set(xscale='log')
    g.map(sns.regplot)
    g.add_legend()
    g.savefig('experiment-epoch-metrics.png')

    plt.clf()

    g = sns.PairGrid(data=data, y_vars=['training_time_s'],
                     x_vars=['max_epoch', 'batch_size', 'num_layers', 'no_nodes'],
                     hue='model_name', palette="vlag")
    # g.set(xscale='log')
    g.map(sns.boxplot)
    g.add_legend()
    g.savefig('experiment-time.png')

    plt.clf()

    g = sns.PairGrid(data=data, y_vars=['avg_energy'],
                     x_vars=['max_epoch', 'batch_size', 'num_layers', 'no_nodes'],
                     hue='model_name', palette="vlag")
    # g.set(xscale='log')
    g.map(sns.boxplot)
    g.set(yticks=[10000, 100000, 1000000], yscale="log")
    g.add_legend()
    g.savefig('experiment-energy.png')

    plt.clf()

    print('===========DONE=========')


def data_accuracy_analysis(data):
    data_cnn = data.loc[(data['model_name'] == "cifar10_cnn") | (data['model_name'] == "mnist_cnn")]
    data_resnet = data.loc[(data['model_name'] == "cifar10_resnet") | (data['model_name'] == "mnist_resnet")]
    print("cnn data")
    desc_cnn = data_cnn.describe()
    print("resnet data")
    desc_res = data_resnet.describe()
    print(desc_res)


if __name__ == '__main__':
    data = pd.read_csv("experiment-data.csv")
    # # data = data.loc[(data['model_name'] == "cifar10_cnn") | (data['model_name'] == "mnist_cnn")]
    # visualise(data)
    # visualise_2(data)
    # visualize_baseline(pd.read_csv("baseline-data.csv"))
    data_accuracy_analysis(data)

    # visualize_3(data)
