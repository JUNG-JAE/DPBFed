import os
from model import CNN
import random
from dataloader import get_dataloader
from sklearn.metrics import f1_score, accuracy_score
import torch
from util import variable_name
from itertools import product
from parameter import WORKER_DATA, LEARNING_MEASURE, SHARD_LIST, UPLOAD_MODEL_NUM, SHARD_NUM
from util import Logger
from MachineLearningUtility import device, load_model, load_worker, model_fraction, add_model, fed_avg


class Worker:
    def __init__(self, *_models, _shard, _worker, _current_round):
        self.models = _models
        self.testloader = get_dataloader("shard" + str(_shard), _worker)
        self.ballot = {}
        self.current_round = _current_round


    def test_global_model(self):
        for index, model in enumerate(self.models):
            model.eval()
            actuals = []
            predictions = []
            with torch.no_grad():
                for data, target in self.testloader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    prediction = output.argmax(dim=1, keepdim=True)
                    actuals.extend(target.view_as(prediction))
                    predictions.extend(prediction)
            actuals, predictions = [i.item() for i in actuals], [i.item() for i in predictions]

            if LEARNING_MEASURE == "f1 score":
                accuracy = round(f1_score(actuals, predictions, average='weighted') * 100, 2)
            else:
                accuracy = round(accuracy_score(actuals, predictions) * 100, 2)

            model_name = variable_name(model)
            Logger("server_logs" + str(self.current_round)).log("Model {0} Accuracy: {1}".format(model_name, accuracy))
            self.ballot[model_name] = accuracy

        max(self.ballot, key=self.ballot.get)
        elected = ''.join([k for k, v in self.ballot.items() if max(self.ballot.values()) == v][0])

        return elected


class GlobalVoting:
    def __init__(self, _round):
        self.round = _round
        self.voting_result = {'previous_global_model': 0, 'current_global_model': 0}

    def global_voting(self):
        if self.round == 1:
            print("Global Round {0}".format(self.round))
            shard1_model = load_model("./model/" + str(1) + "/shard1.pt")
            shard2_model = load_model("./model/" + str(1) + "/shard2.pt")
            shard3_model = load_model("./model/" + str(1) + "/shard3.pt")
            shard4_model = load_model("./model/" + str(1) + "/shard4.pt")
            shard5_model = load_model("./model/" + str(1) + "/shard5.pt")

            avg_model = fed_avg(shard1_model, shard2_model, shard3_model, shard4_model, shard5_model)

            torch.save(avg_model.state_dict(), "./model/" + str(1) + "/aggregation.pt")

            print("Round 1 Model Saved !")

            return

        else:
            previous_global_model = load_model("./model/" + str(int(self.round) - 1) + "/aggregation.pt")

            current_shard1_model = load_model("./model/" + str(self.round) + "/shard1.pt")
            current_shard2_model = load_model("./model/" + str(self.round) + "/shard2.pt")
            current_shard3_model = load_model("./model/" + str(self.round) + "/shard3.pt")
            current_shard4_model = load_model("./model/" + str(self.round) + "/shard4.pt")
            current_shard5_model = load_model("./model/" + str(self.round) + "/shard5.pt")

            current_global_model = fed_avg(current_shard1_model, current_shard2_model, current_shard3_model, current_shard4_model, current_shard5_model)

            for i in range(SHARD_NUM):
                Logger("server_logs" + str(self.round)).log("<<<<<<<<<<<<<<< Shard{0} Voting ... >>>>>>>>>>>>>>>".format(i + 1))
                shard = os.listdir(WORKER_DATA + "shard" + str(i + 1) + "/")
                if ".DS_Store" in shard:
                    shard.remove(".DS_Store")

                for worker in shard:
                    Logger("server_logs" + str(self.round)).log("------------- {0} Voting -------------".format(worker))
                    # worker수가 10개가 넘을 경우 수정 필요.
                    worker_model = Worker(current_global_model, previous_global_model, _shard=i + 1, _worker=int(worker[-1]), _current_round=self.round)
                    vote_result = worker_model.test_global_model()
                    self.voting_result[vote_result] += 1

                Logger("server_logs" + str(self.round)).log("voting result: {0}".format(self.voting_result))

                max(self.voting_result, key=self.voting_result.get)
                elected = ''.join([k for k, v in self.voting_result.items() if max(self.voting_result.values()) == v][-1])
                Logger("server_logs" + str(self.round)).log("elected model: {0}".format(elected))

                if elected == "previous_global_model":
                    Logger("server_logs" + str(self.round)).log("<----------- save previous_global_model ----------->")
                    torch.save(previous_global_model.state_dict(), "./model/" + str(self.round) + "/aggregation.pt")
                elif elected == "current_global_model":
                    Logger("server_logs" + str(self.round)).log("<----------- save current_global_model ----------->")
                    torch.save(current_global_model.state_dict(), "./model/" + str(self.round) + "/aggregation.pt")

            return

