# ----------- System library ----------- #
import os
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import logging

# ----------- Custom library ----------- #
from conf.global_settings import TRANSACTION_PER_MINUTE, TIME, LOG_DIR, SHARD_ID, DATA_TYPE, NUM_OF_WORKER, NUM_OF_MALICIOUS_WORKER, BATCH_SIZE, LEARNING_RATE


def poisson_distribution():
    return [np.random.poisson(TRANSACTION_PER_MINUTE) for _ in range(TIME)]


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_logger(args, global_round):
    create_directory(LOG_DIR + "/" + DATA_TYPE + "/" + args.net + "/logs/")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    file_handler = logging.FileHandler(LOG_DIR + "/" + DATA_TYPE + "/" + args.net + "/logs/" + str(global_round) + "_" + SHARD_ID + "_" + "worker(" + str(NUM_OF_WORKER) + "|" + str(NUM_OF_MALICIOUS_WORKER) + ")" + "_batch(" + str(BATCH_SIZE) + ")" + "_rate(" + str(LEARNING_RATE) + ")" + ".log")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def print_log(logger, msg):
    print(msg)
    logger.info(msg)


def set_global_round(args):
    global_round = 0

    # set global round
    if not os.path.exists(LOG_DIR + "/" + DATA_TYPE + "/" + args.net + "/global_model/G1"):
        print("[ ==================== Global Round: 0 ==================== ]")
        try:
            os.makedirs(LOG_DIR + "/" + DATA_TYPE + "/" + args.net + "/global_model/G1")
        except OSError:
            print('Error: Creating global model directory')
    else:
        rounds = os.listdir(LOG_DIR + "/" + DATA_TYPE + "/" + args.net + "/global_model")
        rounds = [int(round.strip("G")) for round in rounds if round.startswith("G")]
        try:
            current_round = max(rounds)
            os.makedirs(LOG_DIR + "/" + DATA_TYPE + "/" + args.net + "/global_model/G" + str(current_round + 1))
        except OSError:
            print('Error: Creating global model {0:2} directory'.format(current_round + 1))

        global_round = current_round + 1
        print("[ ==================== Global Round: {0:2} ==================== ]".format(global_round))

    return global_round


def get_node_color_and_size(G):
    colors = []
    sizes = []

    for node in G:
        in_degree = G.in_degree[node]
        size = 1000 + 500 * in_degree
        color = 'skyblue' if in_degree >= 2 else 'lightgreen'

        colors.append(color)
        sizes.append(size)

    return colors, sizes


def plot_DAG(args, global_round, tangle):
    create_directory(LOG_DIR + "/" + DATA_TYPE + "/" + args.net + "/graph/")

    # Get edge_list
    frm, to = zip(*[(i, j) for i, edges in tangle.edges.items() for j in edges])

    df = pd.DataFrame({'from': frm, 'to': to})

    # Build the tangle graph
    G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph)

    mapping = {i: (tangle.transaction_pool[i].tx_owner_id, tangle.transaction_pool[i].cumulative_weight) for i in G}
    node_colors, node_sizes = get_node_color_and_size(G)

    nx.draw(G, labels=mapping, node_color=node_colors, node_size=node_sizes, pos=nx.fruchterman_reingold_layout(G))
    plt.title("DAG")
    plt.savefig(LOG_DIR + "/" + DATA_TYPE + "/" + args.net + "/graph/" + str(global_round) + "_" + SHARD_ID + "_" + "worker(" + str(NUM_OF_WORKER) + "|" + str(NUM_OF_MALICIOUS_WORKER) + ")" + "_batch(" + str(BATCH_SIZE) + ")" + "_rate(" + str(LEARNING_RATE) + ")" + ".png")


def central_difference(data):
    differences = []
    for i in range(1, len(data) - 1):
        difference = (data[i + 1] - data[i - 1]) / 2
        differences.append(difference)
    return differences