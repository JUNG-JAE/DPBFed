import warnings
from matplotlib import pyplot as plt
from random import *
import torch
import os
import dag
import util
from util import Logger
from worker import Worker
import parameter as p
import numpy as np

import client_sender as sender
import client_receiver as receiver
import data_indexing as indexing
from round_checker import worker_current_round_chcker

warnings.filterwarnings(action='ignore')

# set current round
current_round = worker_current_round_chcker()

worker_list = []
worker_id_list = []
delayed_worker_list = []

if __name__ == "__main__":

    # create worker
    def create_worker():
        for i in range(0, p.WORKER_NUM):
            worker_id = 'worker' + str(i)
            worker_id_list.append(worker_id)


    def poisson():
        participate_worker_list = []
        time_length = np.random.poisson(p.LAM, p.SIZE)
        for t in time_length:
            if t > p.WORKER_NUM: t = p.WORKER_NUM
            participants = sample(worker_id_list, t)
            participate_worker_list.append(participants)

        return participate_worker_list


    def handler():
        global graph

        participate_worker = poisson()
        print("poisson distribution")
        for participates in participate_worker:
            print(participates, len(participates))

        # poisson distribution used in model weight attack
        # attack_poisson, _ = list(np.histogram(np.random.poisson(p.LAM, p.SIZE), bins=np.array(p.SIZE)))

        # create local worker
        for j in range(0, p.WORKER_NUM):
            worker_list.append(Worker(worker_id=worker_id_list[j], current_round=current_round))
            graph = util.plot_accuracy_graph()

        for shard_round in range(1, p.TOTAL_ROUND + 1):
            train_worker_index = list(range(0, p.WORKER_NUM))
            shuffle(train_worker_index)
            for i in train_worker_index:
                # model weight attack
                if worker_list[i].worker_id in p.POISON_WORKER and worker_list[i].worker_id in participate_worker[shard_round - 1]:
                    if p.ATTACK_TYPE == "MODEL_POISONING":
                        Logger(str(worker_list[i].worker_id)).log("<~~~~~~~~~~~~~~~~~ Model Poisoning Attack ~~~~~~~~~~~~~~~~~>".format(shard_round))
                        worker_list[i].weight_poison_attack()
                    elif p.ATTACK_TYPE == "FGSM":
                        Logger(str(worker_list[i].worker_id)).log("<~~~~~~~~~~~~~~~~~ FGSM Attack ~~~~~~~~~~~~~~~~~>".format(shard_round))
                        worker_list[i].FGSM_attack(training_epochs=p.TRAINING_EPOCH)
                    elif p.ATTACK_TYPE == "PGD":
                        Logger(str(worker_list[i].worker_id)).log("<~~~~~~~~~~~~~~~~~ PGD Attack ~~~~~~~~~~~~~~~~~>".format(shard_round))
                        worker_list[i].PGD_attack(training_epochs=p.TRAINING_EPOCH)
                    elif p.ATTACK_TYPE == "LABEL_FLIP_ATTACK":
                        Logger(str(worker_list[i].worker_id)).log("<~~~~~~~~~~~~~~~~~ Label Flipping Attack ~~~~~~~~~~~~~~~~~>".format(shard_round))
                        worker_list[i].label_flipping_attack(training_epochs=p.TRAINING_EPOCH)
                    elif p.ATTACK_TYPE == "NOISE_ATTACK":
                        Logger(str(worker_list[i].worker_id)).log("<~~~~~~~~~~~~~~~~~ Data Noise Attack ~~~~~~~~~~~~~~~~~>".format(shard_round))
                        worker_list[i].data_noise_attack(training_epochs=p.TRAINING_EPOCH)
                elif worker_list[i].worker_id in participate_worker[shard_round - 1]:  # 학습 부분
                    Logger(str(worker_list[i].worker_id)).log("<=================== Round: {0} ===================>".format(shard_round))
                    worker_list[i].total_training_epoch += p.TRAINING_EPOCH
                    worker_list[i].loacl_learning(training_epochs=p.TRAINING_EPOCH)
                else:
                    Logger(str(worker_list[i].worker_id)).log("<=================== Round: {0} ===================>".format(shard_round))
                    Logger(str(worker_list[i].worker_id)).log(">----------------- {0} was passed -----------------<".format(worker_list[i].worker_id))
                    worker_list[i].round += 1
                    continue

                Logger(str(worker_list[i].worker_id)).log("========== [{0}] transaction invoke ==========".format(worker_list[i].worker_id))
                # tip selection algorithm
                if worker_list[i].worker_id in p.POISON_WORKER:
                    print("Random Tip Selection")
                    dag.generate_transactions(tip_selection_algo="random_tip_selection", payload=worker_list[i].model, local_worker=worker_list[i])
                else:
                    dag.generate_transactions(tip_selection_algo=p.TIP_SELECT_ALGO, payload=worker_list[i].model, local_worker=worker_list[i])
                log_worker_accuracy_prev = worker_list[i].evaluation(worker_list[i].model, True)
                Logger(str(worker_list[i].worker_id)).log("previous worker accuracy: {0}".format(log_worker_accuracy_prev))

                # load two model from blockchain
                weights = worker_list[i].approve_list[worker_list[i].round]
                model1, model2 = None, None
                for j in dag.tangle.transactions:
                    if weights[0] == dag.tangle.transactions[j].tx_id:
                        model1 = dag.tangle.transactions[j].payload
                    elif weights[1] == dag.tangle.transactions[j].tx_id:
                        model2 = dag.tangle.transactions[j].payload

                # model aggregation
                worker_list[i].model_aggregation(model1, model2)
                log_worker_id = worker_list[i].worker_id
                log_worker_accuracy = worker_list[i].evaluation(worker_list[i].model, False)
                Logger(str(worker_list[i].worker_id)).log("[---- {0} model aggregation accuracy: {1:.5f} ----]".format(log_worker_id, log_worker_accuracy))

                log_worker_origin_accuracy = worker_list[i].evaluation(worker_list[i].model, False, "origin")
                Logger(str(worker_list[i].worker_id)).log("[---- {0} model aggregation origin accuracy: {1:.5f} ----]".format(log_worker_id, log_worker_origin_accuracy))

                print(" ")
                graph.add_value(worker_list[i], log_worker_accuracy, log_worker_origin_accuracy)

                worker_list[i].round += 1
        # plot graph & show transaction
        # ===============================================================================================================================
        graph.plot_acc_graph()

        print(" ")

        for i in dag.tangle.transactions:
            dag.tangle.transactions[i].show()

        util.plot_graph()

        Logger("transaction").log("Total Cumulative Weight: {0}".format(dag.tangle.worker_cumulative_weight_dict))

        for worker in worker_list:
            actuals, predictions = worker.test_label_predictions(worker.model, mode="origin")
            plt.figure(figsize=(50, 50))
            util.plot_confusion_matrix(actuals, predictions, normalize=True, classes=p.label, title=worker.worker_id)

            plt.savefig("./graph/" + str(worker.worker_id) + "_confustion_matrix.png")
            # plt.show()

        # 1. send shard model and get another shard model
        # 2. send shard image index to server, this means worker migrate to other shards
        # ===============================================================================================================================

        dag.save_shard_global_model(current_round)

        # if p.MULTI_UPLOAD:
        #     # 여러개의 shard model을 업로드 할 경우
        #     for i in range(p.UPLOAD_MODEL_NUM):
        #         sender.send_file(p.SERVER_HOST, p.SERVER_PORT, p.SAVE_SHARD_MODEL_PATH + str(current_round) + "/" + p.SHARD_ID + "_" + str(i) + ".pt")
        # else:
        #     # 한개의 shard model을 업로드 할 경우
        sender.send_file(p.SERVER_HOST, p.SERVER_PORT, p.SAVE_SHARD_MODEL_PATH + str(current_round) + "/" + p.SHARD_ID + ".pt")

        receiver.runReceiver()

        path = p.SAVE_SHARD_MODEL_PATH + str(current_round) + "/"
        # file_list = os.listdir(path)
        # file_list = ["shard1.pt", "shard2.pt", "shard3.pt", "shard4.pt", "shard5.pt"]
        file_list = ["shard1.pt", "shard2.pt"]
        if ".DS_Store" in file_list:
            file_list.remove(".DS_Store")
        if "aggregation.pt" in file_list:
            file_list.remove("aggregation.pt")

        print("file_list: {}".format(file_list))

        for worker in worker_list:
            acc_list = {}
            for model_path in file_list:
                path = "./model/" + str(current_round) + "/" + model_path
                worker.model.load_state_dict(torch.load(path), strict=False)
                accuracy = worker.evaluation(worker.model)
                acc_list[str(model_path).replace(".pt", "")] = accuracy

            max(acc_list, key=acc_list.get)
            max_acc_shard = [k for k, v in acc_list.items() if max(acc_list.values()) == v]
            if len(max_acc_shard) >= 1:
                max_acc_shard = choice(max_acc_shard)
            characters = "['']"
            max_acc_shard = ''.join(x for x in max_acc_shard if x not in characters)
            p.shard_list[max_acc_shard].append(worker.worker_id)

        Logger("transaction").log("next shard status: {0}".format(p.shard_list))

        train_index_info = indexing.worker_to_index("train", p.shard_list, )
        test_index_info = indexing.worker_to_index("test", p.shard_list, )

        PAYLOAD = {
            "train": train_index_info,
            "test": test_index_info
        }

        file = open("./migrate/" + p.SHARD_ID + ".txt", 'w')
        file.write(str(PAYLOAD))
        file.close()

        # for worker migration
        # sender.send_file(p.MIGRATION_SERVER_HOST, p.MIGRATION_SERVER_PORT, p.SAVE_MIGRATION_INFO_PATH)
        # receiver.runReceiver(True)
        
        # data_classifier.migrate_worker()

    # main handler
    # ===============================================================================================================================
    create_worker()
    dag.generate_transactions(initial=True, initial_count=1)
    handler()
