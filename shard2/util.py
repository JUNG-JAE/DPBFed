import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from logging import handlers
import logging
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cityblock, cosine
import torch

import dag
import parameter as p


# handler settings
class Logger:
    def __init__(self, log_file_name):
        self.log_formatter = logging.Formatter('%(asctime)s,%(message)s')
        self.handler = handlers.TimedRotatingFileHandler(filename='logs/' + str(log_file_name), when='midnight',
                                                         interval=1, encoding='utf-8')
        self.handler.setFormatter(self.log_formatter)
        self.handler.suffix = "%Y%m%d"

        # logger set
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.handler)

    def get_logger(self):
        return self.logger

    def log(self, message):
        print(message)
        self.logger.info(str(message))
        for hdlr in self.logger.handlers[:]:
            self.logger.removeHandler(hdlr)


class plot_accuracy_graph:
    def __init__(self):
        self.worker_dict = {}

        for i in range(0, p.WORKER_NUM):
            worker_id = 'worker' + str(i)
            self.worker_dict[worker_id] = []

    def add_value(self, worker, accuracy, origin_accuracy):
        self.worker_dict[worker.worker_id].append([worker.total_training_epoch, accuracy, origin_accuracy])

    def plot_acc_graph(self):
        plt.figure(figsize=(10, 10))
        plt.tight_layout()

        x_axis_list = []
        y1_axis_list = []
        y2_axis_list = []

        for worker in self.worker_dict:
            x_axis = [0]
            y1_axis = [0]
            y2_axis = [0]
            for x, y1, y2 in self.worker_dict[worker]:
                x_axis.append(x)
                y1_axis.append(y1)
                y2_axis.append(y2)
            x_axis_list.append(x_axis)
            y1_axis_list.append(y1_axis)
            y2_axis_list.append(y2_axis)

        for index, worker in enumerate(self.worker_dict):
            sub_acc_graph = plt.subplot(4, 4, int(worker[-1:]) + 1)
            print("X: ", x_axis_list[index])
            print("Y1: ", y1_axis_list[index])
            print("Y2: ", y2_axis_list[index])

            file = open("./acc_graph_data/" + str(worker) + "_data.txt", "w")
            file.write(str(worker)+"\n")
            file.write((str(x_axis_list[index])+"\n"))
            file.write((str(y1_axis_list[index])+"\n"))
            file.write((str(y2_axis_list[index]) + "\n"))
            file.close()

            sub_acc_graph.plot(x_axis_list[index], y1_axis_list[index], label=worker, marker='o')
            sub_acc_graph.plot(x_axis_list[index], y2_axis_list[index], label=worker, marker='o')

            sub_acc_graph.set_title(worker)
            sub_acc_graph.set_xlabel("epoch")
            sub_acc_graph.set_ylabel("local aggregation accuracy")
            plt.tight_layout()
            plt.xticks(np.arange(0, (p.TRAINING_EPOCH*p.TOTAL_ROUND)+1, p.TRAINING_EPOCH))
            plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])


        plt.savefig("./graph/" + str(p.SHARD_ID) + '.png')
        # plt.show()

# plot graph
def plot_graph():
    edge_list = dag.tangle.edges
    frm = []
    to = []

    for i in edge_list.keys():
        for j in edge_list[i]:
            frm.append(i)
            to.append(j)

    df = pd.DataFrame({'from': frm, 'to': to})

    # Build the tangle graph
    G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph)

    j = 65
    mapping = {}
    cols = []
    size = []

    for i in G:
        wt = dag.tangle.transactions[i].cumulative_weight
        mapping[i] = (chr(j), wt)
        j = j + 1
        size.append(1000 + 500 * G.in_degree[i])
        if G.in_degree[i] >= 2:
            cols.append('skyblue')
        else:
            cols.append('lightgreen')

    nx.draw(G, labels=mapping, node_color=cols, node_size=size, pos=nx.fruchterman_reingold_layout(G))
    plt.figure(1)
    plt.title("Tangle")
    plt.savefig("./graph/" + str(p.SHARD_ID) + '_dag.png')
    # plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("{0} Normalized confusion matrix".format(title))
    else:
        print('{0} Confusion matrix, without normalization'.format(title))

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),

           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

np.set_printoptions(precision=2)


def gaussian_distribution(x, mean, sigma):
    return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


def vector_similarity(model1, model2):
    layer1 = 1 - cosine(model1.layer1[0].weight.data.numpy().reshape(-1, ), model2.layer1[0].weight.data.numpy().reshape(-1, ))
    layer2 = 1 - cosine(model1.layer2[0].weight.data.numpy().reshape(-1, ), model2.layer2[0].weight.data.numpy().reshape(-1, ))
    layer3 = 1 - cosine(model1.layer3[0].weight.data.numpy().reshape(-1, ), model2.layer3[0].weight.data.numpy().reshape(-1, ))
    fc1 = 1 - cosine(model1.fc1.weight.data.numpy().reshape(-1, ), model2.fc1.weight.data.numpy().reshape(-1, ))
    fc2 = 1 - cosine(model1.fc2.weight.data.numpy().reshape(-1, ), model2.fc2.weight.data.numpy().reshape(-1, ))
    
    return (layer1 + layer2 + layer3 + fc1 + fc2) / 5
        

def quantize(x):
    n=32
    x=x.float()
    x_norm=torch.norm(x,p=float('inf'))
    
    sgn_x=((x>0).float()-0.5)*2
    
    p=torch.div(torch.abs(x),x_norm)
    renormalize_p=torch.mul(p,n)
    floor_p=torch.floor(renormalize_p)
    compare=torch.rand_like(floor_p)
    final_p=renormalize_p-floor_p
    margin=(compare < final_p).float()
    xi=(floor_p+margin)/n
    
    Tilde_x=x_norm*sgn_x*xi
    
    return Tilde_x
