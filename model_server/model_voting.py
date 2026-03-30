import os
from model import CNN
import random
from dataloader import get_dataloader
from sklearn.metrics import f1_score, accuracy_score
import torch
from util import variable_name
from itertools import product
from parameter import WORKER_DATA, LEARNING_MEASURE, SHARD_LIST, UPLOAD_MODEL_NUM
from util import Logger
from MachineLearningUtility import device, load_model, load_worker, model_fraction, add_model


class Worker:
    def __init__(self, *_models, _shard, _worker, _current_round):
        self.models = _models
        self.testloader = get_dataloader(str(_shard), _worker)
        self.ballot = {}
        self.current_round = _current_round

    """
    def test_global_model(self, model_id_list):
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

            if LEARNING_MEASURE == "accuracy":
                accuracy = round(accuracy_score(actuals, predictions) * 100, 2)
            elif LEARNING_MEASURE == "f1 score":
                accuracy = round(f1_score(actuals, predictions, average='weighted') * 100, 2)

            # model_name = variable_name(model)
            # print("Model {0} Accuracy: {1}".format(model_id_list[index], accuracy))
            Logger("server_logs" + str(self.current_round)).log("Model {0} Accuracy: {1}".format(model_id_list[index], accuracy))
            self.ballot[model_id_list[index]] = accuracy

        max(self.ballot, key=self.ballot.get)
        elected = ''.join([k for k, v in self.ballot.items() if max(self.ballot.values()) == v][0])

        return elected
    """

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


"""
class Voting:
    def __init__(self, _current_round):
        self.first_voting = True
        self.voting_committee = []
        self.first_combination_models = []
        self.first_combination_model_names = []
        self.current_round = _current_round
        self.SAVE_MODEL_PATH = "./model/" + str(self.current_round) + "/"

    def model_voter(self):
        if "g1.pt" in os.listdir(self.SAVE_MODEL_PATH):
            self.first_voting = False

        if self.first_voting:
            # print("first voting")
            Logger("server_logs" + str(self.current_round)).log("first voting")
            first_shard_models = []
            first_shard_models_name = []
            second_shard_models = []
            second_shard_models_name = []
            input_models = []
            input_models_name = []
            voting_result = {}

            # 샤드 목록중 랜덤하게 2개의 샤드를 선택하고 선택된 샤드는 리스트에서 지운다.
            random_shards = random.sample(SHARD_LIST, 2)
            SHARD_LIST.remove(random_shards[0])
            SHARD_LIST.remove(random_shards[1])
            self.voting_committee.append(random_shards[0])
            self.voting_committee.append(random_shards[1])

            # 1.선택된 샤드에서 업데이트된 모델을 가져오고 모델의 이름을 생성한다.
            for index in range(UPLOAD_MODEL_NUM):
                # first shard
                model1 = load_model(self.SAVE_MODEL_PATH + random_shards[0] + "_" + str(index) + ".pt")
                model_fraction(model1, 1, 5)
                # first_shard_models.append(load_model(self.SAVE_MODEL_PATH + random_shards[0] + "_" + str(index) + ".pt"))
                first_shard_models.append(model1)
                first_shard_models_name.append(random_shards[0] + "_" + str(index))

                # second shard
                model2 = load_model(self.SAVE_MODEL_PATH + random_shards[1] + "_" + str(index) + ".pt")
                model_fraction(model2, 1, 5)
                # second_shard_models.append(load_model(self.SAVE_MODEL_PATH + random_shards[1] + "_" + str(index) + ".pt"))
                second_shard_models.append(model2)
                second_shard_models_name.append(random_shards[1] + "_" + str(index))

            # model combination을 위해 input_models 리스트에 랜덤으로 선택된 샤드들을 넣는다. -> [[], []] 2차원 리스트
            input_models.append(first_shard_models)
            input_models.append(second_shard_models)
            input_models_name.append(first_shard_models_name)
            input_models_name.append(second_shard_models_name)

            # 샤드 모델 combination
            combination_models = list(product(*input_models))
            combination_models_name = list(product(*input_models_name))
            # print("Selected Shard Models: {0}".format(input_models_name))
            # print("Combination model list: {0} length {1}".format(combination_models_name, len(combination_models)))
            Logger("server_logs" + str(self.current_round)).log("Selected Shard Models: {0}".format(input_models_name))
            Logger("server_logs" + str(self.current_round)).log("Combination model list: {0} length {1}".format(combination_models_name, len(combination_models)))

            # 조합된 모델 FedAsyncAvg
            for models in combination_models:
                # model_fraction(models[0], 1, 5)
                # model_fraction(models[1], 1, 5)
                model = add_model(models[0], models[1])
                model_fraction(model, 5, 2)

                self.first_combination_models.append(model)

            # 조합된 모델 이름 생성
            for names in combination_models_name:
                model_name = names[0] + "+" + names[1]

                self.first_combination_model_names.append(model_name)
                voting_result[model_name] = 0

            # 해당 샤드의 worker 수를 가져오기 위해 사용
            shard1_worker_length = len(load_worker(random_shards[0]))
            shard2_worker_length = len(load_worker(random_shards[1]))

            # print("Voting Committee: {0}".format(self.voting_committee))
            Logger("server_logs" + str(self.current_round)).log("Voting Committee: {0}".format(self.voting_committee))
            print(len(self.voting_committee))

            # print("Voting Shard: {0}".format(random_shards[0]))
            Logger("server_logs" + str(self.current_round)).log("Voting Shard: {0}".format(random_shards[0]))
            for worker_id in range(shard1_worker_length):
                # print("=============== Worker{0} ===============".format(worker_id))
                Logger("server_logs" + str(self.current_round)).log("=============== Worker{0} ===============".format(worker_id))
                worker = Worker(*self.first_combination_models, _shard=random_shards[0], _worker=worker_id, _current_round=self.current_round)
                elect_result = worker.test_global_model(self.first_combination_model_names)
                voting_result[elect_result] += 1
                # print("<----- elected: {0} ----->\n".format(elect_result))
                Logger("server_logs" + str(self.current_round)).log("<----- elected: {0} ----->\n".format(elect_result))

            # print("Voting Shard: {0}".format(random_shards[1]))
            Logger("server_logs" + str(self.current_round)).log("Voting Shard: {0}".format(random_shards[1]))
            for worker_id in range(shard2_worker_length):
                # print("=============== Worker{0} ===============".format(worker_id))
                Logger("server_logs" + str(self.current_round)).log("=============== Worker{0} ===============".format(worker_id))
                worker = Worker(*self.first_combination_models, _shard=random_shards[1], _worker=worker_id, _current_round=self.current_round)
                elect_result = worker.test_global_model(self.first_combination_model_names)
                voting_result[elect_result] += 1
                # print("<----- elected: {0} ----->\n".format(elect_result))
                Logger("server_logs" + str(self.current_round)).log("<----- elected: {0} ----->\n".format(elect_result))

            # print("After Voting: {0}".format(voting_result))
            Logger("server_logs" + str(self.current_round)).log("After Voting: {0}".format(voting_result))
            max(voting_result, key=voting_result.get)
            elected = ''.join([k for k, v in voting_result.items() if max(voting_result.values()) == v][0])
            # print(elected)
            Logger("server_logs" + str(self.current_round)).log(elected)

            elected_model_index = self.first_combination_model_names.index(elected)
            elected_model = self.first_combination_models[elected_model_index]
            using_model = elected.split("+")
            model_fraction(elected_model, 2, 5)

            torch.save(elected_model.state_dict(), "./model/" + str(self.current_round) + "/g1.pt")
            first_using_model = load_model(self.SAVE_MODEL_PATH + using_model[0] + ".pt")
            second_using_model = load_model(self.SAVE_MODEL_PATH + using_model[1] + ".pt")
            torch.save(first_using_model.state_dict(), "./model/" + str(self.current_round) + "/" + using_model[0][:6] + ".pt")
            torch.save(second_using_model.state_dict(), "./model/" + str(self.current_round) + "/" + using_model[1][:6] + ".pt")

            return
        else:
            shard_models = []
            shard_models_name = []
            input_models = []
            input_models_name = []
            ballot_combination_model = []
            ballot_combination_model_names = []
            voting_result = {}

            # print("model voting")
            Logger("server_logs" + str(self.current_round)).log("model voting")
            model_list = os.listdir(self.SAVE_MODEL_PATH)

            # 이전 투표를 통해 얻은 글로벌 모델을 가져온다.
            if "g3.pt" in model_list:
                # print("load model g3")
                Logger("server_logs" + str(self.current_round)).log("load model g3")
                pre_model = [load_model(self.SAVE_MODEL_PATH + "g3.pt")]
                pre_model_name = ["g3"]
                save_model_name = "g4"
            elif "g2.pt" in model_list:
                # print("load model g2")
                Logger("server_logs" + str(self.current_round)).log("load model g2")
                pre_model = [load_model(self.SAVE_MODEL_PATH + "g2.pt")]
                pre_model_name = ["g2"]
                save_model_name = "g3"
            else:
                # print("load model g1")
                Logger("server_logs" + str(self.current_round)).log("load model g1")
                pre_model = [load_model(self.SAVE_MODEL_PATH + "g1.pt")]
                pre_model_name = ["g1"]
                save_model_name = "g2"

            # 이전에 투표에서 참여한 모델을 제외하고 랜덤하게 샤드를 선택한다.
            random_shards = random.sample(SHARD_LIST, 1)
            SHARD_LIST.remove(random_shards[0])
            self.voting_committee.append(random_shards[0])

            # 이전 모델과 새롭게 선택된 샤드의 모델을 combination하기 위해 리스트에 이전 모델과 모델 이름을 추가한다.
            input_models.append(pre_model)
            input_models_name.append(pre_model_name)

            # 1.선택된 샤드에서 업데이트된 모델을 가져오고 모델의 이름을 생성한다.
            for index in range(UPLOAD_MODEL_NUM):
                # first shard
                model1 = load_model(self.SAVE_MODEL_PATH + random_shards[0] + "_" + str(index) + ".pt")
                model_fraction(model1, 1, 5)
                # shard_models.append(load_model(self.SAVE_MODEL_PATH + random_shards[0] + "_" + str(index) + ".pt"))
                shard_models.append(model1)
                shard_models_name.append(random_shards[0] + "_" + str(index))

            # model combination을 위해 input_models 리스트에 랜덤으로 선택된 샤드들을 넣는다. -> [[], []] 2차원 리스트
            input_models.append(shard_models)
            input_models_name.append(shard_models_name)

            # 샤드 모델 combination
            combination_models = list(product(*input_models))
            combination_models_name = list(product(*input_models_name))
            # print("Selected Shard Models: {0}".format(input_models_name))
            # print("Combination model list: {0} length {1}".format(combination_models_name, len(combination_models)))
            Logger("server_logs" + str(self.current_round)).log("Selected Shard Models: {0}".format(input_models_name))
            Logger("server_logs" + str(self.current_round)).log("Combination model list: {0} length {1}".format(combination_models_name, len(combination_models)))

            # 조합된 모델 FedAsyncAvg
            for models in combination_models:
                # model_fraction(models[1], 1, 5)
                model = add_model(models[0], models[1])
                model_fraction(model, 5, len(self.voting_committee))

                ballot_combination_model.append(model)

            # 조합된 모델 이름 생성
            for names in combination_models_name:
                model_name = names[0] + "+" + names[1]

                ballot_combination_model_names.append(model_name)
                voting_result[model_name] = 0

            # print("Voting Committee: {0}".format(self.voting_committee))
            Logger("server_logs" + str(self.current_round)).log("Voting Committee: {0}".format(self.voting_committee))
            print(len(self.voting_committee))
            for shard in self.voting_committee:
                # voting committee에 있는 각 샤드들의 worker 수를 가져온다.
                worker_length = len(load_worker(shard))

                # print("Voting Shard: {0}".format(shard))
                Logger("server_logs" + str(self.current_round)).log("Voting Shard: {0}".format(shard))
                for worker_id in range(worker_length):
                    # print("=============== Worker{0} ===============".format(worker_id))
                    Logger("server_logs" + str(self.current_round)).log("=============== Worker{0} ===============".format(worker_id))
                    worker = Worker(*ballot_combination_model, _shard=random_shards[0], _worker=worker_id, _current_round=self.current_round)
                    elect_result = worker.test_global_model(ballot_combination_model_names)
                    voting_result[elect_result] += 1
                    # print("<----- elected: {0} ----->\n".format(elect_result))
                    Logger("server_logs" + str(self.current_round)).log("<----- elected: {0} ----->\n".format(elect_result))

            # print("After Voting: {0}".format(voting_result))
            Logger("server_logs" + str(self.current_round)).log("After Voting: {0}".format(voting_result))
            max(voting_result, key=voting_result.get)
            elected = ''.join([k for k, v in voting_result.items() if max(voting_result.values()) == v][0])
            # print(elected)
            Logger("server_logs" + str(self.current_round)).log(elected)

            elected_model_index = ballot_combination_model_names.index(elected)
            using_model_name = elected.split("+")
            elected_model = ballot_combination_model[elected_model_index]
            model_fraction(elected_model, len(self.voting_committee), 5)

            torch.save(elected_model.state_dict(), "./model/" + str(self.current_round) + "/" + save_model_name + ".pt")

            using_model = load_model(self.SAVE_MODEL_PATH + using_model_name[1] + ".pt")
            torch.save(using_model.state_dict(), "./model/" + str(self.current_round) + "/" + using_model_name[1][:6] + ".pt")

            return
"""


class ModelCombinator:
    def __init__(self, _round, _mode, _model=None):
        self.round = _round
        self.mode = _mode

        self.previous_A_previous_B = None
        self.previous_A_current_B = None
        self.current_A_previous_B = None
        self.current_A_current_B = None

        self.global_model_previous_model = None
        self.global_model_current_model = None

    def combinator(self):
        if self.mode == "A+B":
            previous_round_model_A = load_model("./model/"+str(self.round-1)+"/shard1.pt")
            current_round_model_A = load_model("./model/"+str(self.round)+"/shard1.pt")
            previous_round_model_B = load_model("./model/"+str(self.round-1)+"/shard2.pt")
            current_round_model_B = load_model("./model/"+str(self.round)+"/shard2.pt")

            model_fraction(previous_round_model_A, 1, 5)
            model_fraction(current_round_model_A, 1, 5)
            model_fraction(previous_round_model_B, 1, 5)
            model_fraction(current_round_model_B, 1, 5)

            self.previous_A_previous_B = add_model(previous_round_model_A, previous_round_model_B)
            model_fraction(self.previous_A_previous_B, 5, 2)
            self.previous_A_current_B = add_model(previous_round_model_A, current_round_model_B)
            model_fraction(self.previous_A_current_B, 5, 2)
            self.current_A_previous_B = add_model(current_round_model_A, previous_round_model_B)
            model_fraction(self.current_A_previous_B, 5, 2)
            self.current_A_current_B = add_model(current_round_model_A, current_round_model_B)
            model_fraction(self.current_A_current_B, 5, 2)

            return self.previous_A_previous_B, self.previous_A_current_B, self.current_A_previous_B, self.current_A_current_B

        elif self.mode == "A+B+C":  # G1 + C
            previous_global_model = load_model("./model/" + str(self.round) + "/g1.pt")
            previous_round_model_C = load_model("./model/" + str(self.round - 1) + "/shard3.pt")
            current_round_model_C = load_model("./model/" + str(self.round) + "/shard3.pt")

            model_fraction(previous_round_model_C, 1, 5)
            model_fraction(current_round_model_C, 1, 5)

            self.global_model_previous_model = add_model(previous_global_model, previous_round_model_C)
            model_fraction(self.global_model_previous_model, 5, 3)
            self.global_model_current_model = add_model(previous_global_model, current_round_model_C)
            model_fraction(self.global_model_current_model, 5, 3)

            return self.global_model_previous_model, self.global_model_current_model

        elif self.mode == "A+B+C+D":  # G2 + D
            previous_global_model = load_model("./model/" + str(self.round) + "/g2.pt")
            previous_round_model_D = load_model("./model/" + str(self.round - 1) + "/shard4.pt")
            current_round_model_D = load_model("./model/" + str(self.round) + "/shard4.pt")

            model_fraction(previous_round_model_D, 1, 5)
            model_fraction(current_round_model_D, 1, 5)

            self.global_model_previous_model = add_model(previous_global_model, previous_round_model_D)
            model_fraction(self.global_model_previous_model, 5, 4)
            self.global_model_current_model = add_model(previous_global_model, current_round_model_D)
            model_fraction(self.global_model_current_model, 5, 4)

            return self.global_model_previous_model, self.global_model_current_model

        elif self.mode == "A+B+C+D+E":  # G2 + D
            previous_global_model = load_model("./model/" + str(self.round) + "/g3.pt")
            previous_round_model_E = load_model("./model/" + str(self.round - 1) + "/shard5.pt")
            current_round_model_E = load_model("./model/" + str(self.round) + "/shard5.pt")

            model_fraction(previous_round_model_E, 1, 5)
            model_fraction(current_round_model_E, 1, 5)

            self.global_model_previous_model = add_model(previous_global_model, previous_round_model_E)
            model_fraction(self.global_model_previous_model, 5, 5)
            self.global_model_current_model = add_model(previous_global_model, current_round_model_E)
            model_fraction(self.global_model_current_model, 5, 5)

            return self.global_model_previous_model, self.global_model_current_model


    def aggregator(self, elected_model):
        if self.mode == "A+B":
            if elected_model == "model1":
                model_fraction(self.previous_A_previous_B, 2, 5)
                torch.save(self.previous_A_previous_B.state_dict(), "./model/" + str(self.round) + "/g1.pt")
            elif elected_model == "model2":
                model_fraction(self.previous_A_current_B, 2, 5)
                torch.save(self.previous_A_current_B.state_dict(), "./model/" + str(self.round) + "/g1.pt")
            elif elected_model == "model3":
                model_fraction(self.current_A_previous_B, 2, 5)
                torch.save(self.current_A_previous_B.state_dict(), "./model/" + str(self.round) + "/g1.pt")
            elif elected_model == "model4":
                model_fraction(self.current_A_current_B, 2, 5)
                torch.save(self.current_A_current_B.state_dict(), "./model/" + str(self.round) + "/g1.pt")

            return

        elif self.mode == "A+B+C":  # G1 + C
            if elected_model == "model1":
                model_fraction(self.global_model_previous_model, 3, 5)
                torch.save(self.global_model_previous_model.state_dict(), "./model/" + str(self.round) + "/g2.pt")
            elif elected_model == "model2":
                model_fraction(self.global_model_current_model, 3, 5)
                torch.save(self.global_model_current_model.state_dict(), "./model/" + str(self.round) + "/g2.pt")

            return

        elif self.mode == "A+B+C+D":  # G2 + D
            if elected_model == "model1":
                model_fraction(self.global_model_previous_model, 4, 5)
                torch.save(self.global_model_previous_model.state_dict(), "./model/" + str(self.round) + "/g3.pt")
            elif elected_model == "model2":
                model_fraction(self.global_model_current_model, 4, 5)
                torch.save(self.global_model_current_model.state_dict(), "./model/" + str(self.round) + "/g3.pt")

            return

        elif self.mode == "A+B+C+D+E":  # G3 + E
            if elected_model == "model1":
                model_fraction(self.global_model_previous_model, 5, 5)
                torch.save(self.global_model_previous_model.state_dict(), "./model/" + str(self.round) + "/aggregation.pt")
            elif elected_model == "model2":
                model_fraction(self.global_model_current_model, 5, 5)
                torch.save(self.global_model_current_model.state_dict(), "./model/" + str(self.round) + "/aggregation.pt")

            return

class Voting:
    def __init__(self, _round, _global_model_type):
        self.global_model_type = _global_model_type
        self.round = _round
        self.combinator_class = None

        if self.global_model_type == "A+B":
            self.voting_result = {'model1' : 0, 'model2' : 0, 'model3' : 0, 'model4' : 0}
        else:
            self.voting_result = {'model1': 0, 'model2': 0}

    def handler(self):
        if self.round == 1:
            A = load_model("./model/1/shard1.pt")
            B = load_model("./model/1/shard2.pt")
            C = load_model("./model/1/shard3.pt")
            D = load_model("./model/1/shard4.pt")
            E = load_model("./model/1/shard5.pt")

            model_fraction(A, 1, 5)
            model_fraction(B, 1, 5)
            model_fraction(C, 1, 5)
            model_fraction(D, 1, 5)
            model_fraction(E, 1, 5)

            model = add_model(A, B, C, D, E)

            torch.save(model.state_dict(), "./model/" + str(self.round) + "/aggregation.pt")

            return
        else:
            if self.global_model_type == "A+B":
                self.combinator_class = ModelCombinator(self.round, self.global_model_type)
                model1, model2, model3, model4 = self.combinator_class.combinator()

                for i in range(2):
                    Logger("server_logs" + str(self.round)).log("<<<<<<<<<<<<<<< Shard{0} Voting ... >>>>>>>>>>>>>>>".format(i + 1))
                    shard = os.listdir(WORKER_DATA + "shard" + str(i+1) + "/")
                    if ".DS_Store" in shard:
                        shard.remove(".DS_Store")

                    for worker in shard:
                        Logger("server_logs" + str(self.round)).log("------------- {0} Voting -------------".format(worker))
                        worker_model = Worker(model1, model2, model3, model4, _shard="shard"+str(i + 1), _worker=int(worker[-1]), _current_round=self.round)
                        vote_result = worker_model.test_global_model()
                        self.voting_result[vote_result] += 1

            elif self.global_model_type == "A+B+C":
                self.combinator_class = ModelCombinator(self.round, self.global_model_type)
                model1, model2 = self.combinator_class.combinator()

                for i in range(3):
                    Logger("server_logs" + str(self.round)).log("<<<<<<<<<<<<<<< Shard{0} Voting ... >>>>>>>>>>>>>>>".format(i + 1))
                    shard = os.listdir(WORKER_DATA + "shard" + str(i + 1) + "/")
                    if ".DS_Store" in shard:
                        shard.remove(".DS_Store")

                    for worker in shard:
                        Logger("server_logs" + str(self.round)).log("------------- {0} Voting -------------".format(worker))
                        worker_model = Worker(model1, model2, _shard="shard"+str(i + 1), _worker=int(worker[-1]), _current_round=self.round)
                        vote_result = worker_model.test_global_model()
                        self.voting_result[vote_result] += 1

            elif self.global_model_type == "A+B+C+D":
                self.combinator_class = ModelCombinator(self.round, self.global_model_type)
                model1, model2 = self.combinator_class.combinator()

                for i in range(4):
                    Logger("server_logs" + str(self.round)).log("<<<<<<<<<<<<<<< Shard{0} Voting ... >>>>>>>>>>>>>>>".format(i + 1))
                    shard = os.listdir(WORKER_DATA + "shard" + str(i + 1) + "/")
                    if ".DS_Store" in shard:
                        shard.remove(".DS_Store")

                    for worker in shard:
                        Logger("server_logs" + str(self.round)).log("------------- {0} Voting -------------".format(worker))
                        worker_model = Worker(model1, model2, _shard="shard"+str(i + 1), _worker=int(worker[-1]), _current_round=self.round)
                        vote_result = worker_model.test_global_model()
                        self.voting_result[vote_result] += 1

            elif self.global_model_type == "A+B+C+D+E":
                self.combinator_class = ModelCombinator(self.round, self.global_model_type)
                model1, model2 = self.combinator_class.combinator()

                for i in range(5):
                    Logger("server_logs" + str(self.round)).log("<<<<<<<<<<<<<<< Shard{0} Voting ... >>>>>>>>>>>>>>>".format(i + 1))
                    shard = os.listdir(WORKER_DATA + "shard" + str(i + 1) + "/")
                    if ".DS_Store" in shard:
                        shard.remove(".DS_Store")

                    for worker in shard:
                        Logger("server_logs" + str(self.round)).log("------------- {0} Voting -------------".format(worker))
                        worker_model = Worker(model1, model2, _shard="shard"+str(i + 1), _worker=int(worker[-1]), _current_round=self.round)
                        vote_result = worker_model.test_global_model()
                        self.voting_result[vote_result] += 1

            Logger("server_logs" + str(self.round)).log("voting result: {0}".format(self.voting_result))

            max(self.voting_result, key=self.voting_result.get)
            elected = ''.join([k for k, v in self.voting_result.items() if max(self.voting_result.values()) == v][-1])
            Logger("server_logs" + str(self.round)).log("elected model: {0}".format(elected))
            self.combinator_class.aggregator(elected)

            return
