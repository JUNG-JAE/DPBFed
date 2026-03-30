import server_receiver as receiver
import server_send as sender
import parameter as p
from round_checker import current_round_checker
from mitigate_update import Voting
import torch
from util import Logger
from MachineLearningUtility import load_model, mitigate_fed_avg, test_label_predictions, fed_avg
from sklearn.metrics import accuracy_score
from parameter import GLOBAL_MODEL_ALPHA

current_round = current_round_checker()
receiver.runServer()

"""
# global blockchain의 이전 모델과 업데이트된 모델을 비교투표 하여 global model 생성
if current_round == 1:
    # print("=========== Round 1 Model Aggregation ===========")
    Logger("server_logs" + str(current_round)).log("=========== Round 1 Model Aggregation ===========")
    Voting(1, "A+B+C+D+E").handler()
else:
    # print("=========== s1+s2 voting ===========")
    Logger("server_logs" + str(current_round)).log("=========== s1+s2 voting ===========")
    Voting(current_round, "A+B").handler()

    # print("========== s1+s2+s3 voting ==========")
    Logger("server_logs" + str(current_round)).log("========== s1+s2+s3 voting ==========")
    Voting(current_round, "A+B+C").handler()

    # print("========= s1+s2+s3+s4 voting =========")
    Logger("server_logs" + str(current_round)).log("========= s1+s2+s3+s4 voting =========")
    Voting(current_round, "A+B+C+D").handler()

    # print("======== s1+s2+s3+s4+s5 voting ========")
    Logger("server_logs" + str(current_round)).log("======== s1+s2+s3+s4+s5 voting ========")
    Voting(current_round, "A+B+C+D+E").handler()

    # # mitigate model update
    # previous_model = load_model("./model/" + str(current_round - 1) + "/aggregation.pt")
    # current_model = load_model("./model/" + str(current_round) + "/aggregation.pt")
    #
    # mitigate_model = mitigate_fed_avg(previous_model, current_model, alpha=0.5)
    # torch.save(mitigate_model.state_dict(), "./model/" + str(current_round) + "/aggregation.pt")
"""
# ================================================================================================================
"""
# 각 shard로부터 3개의 모델이 업로드되면 투표를 통해 모델을 선택한 후 가장 좋은 모델을 이용하여 global model을 생성한다.
print("================== MODEL VOTING ==================")
handler = Voting(current_round)
handler.model_voter()
handler.model_voter()
handler.model_voter()
handler.model_voter()

shard1 = load_model("./model/" + str(current_round) + "/shard1.pt")
shard2 = load_model("./model/" + str(current_round) + "/shard2.pt")
shard3 = load_model("./model/" + str(current_round) + "/shard3.pt")
shard4 = load_model("./model/" + str(current_round) + "/shard4.pt")
shard5 = load_model("./model/" + str(current_round) + "/shard5.pt")

print("aggregation model")
avg = fed_avg(shard1, shard2, shard3, shard4, shard5)
torch.save(avg.state_dict(), "./model/" + str(current_round) + "/aggregation.pt")
"""
# ================================================================================================================
"""
# 동기적으로 투표 없이 FedAvg
shard1 = load_model("./model/" + str(current_round) + "/shard1.pt")
shard2 = load_model("./model/" + str(current_round) + "/shard2.pt")
shard3 = load_model("./model/" + str(current_round) + "/shard3.pt")
shard4 = load_model("./model/" + str(current_round) + "/shard4.pt")
shard5 = load_model("./model/" + str(current_round) + "/shard5.pt")

print("aggregation model")
avg = fed_avg(shard1, shard2, shard3, shard4, shard5)
torch.save(avg.state_dict(), "./model/" + str(current_round) + "/aggregation.pt")
"""
# ================================================================================================================
"""
# global model을 생성한 후 global model에 대해서 모든 worker들이 투표를 한다.
handler = GlobalVoting(current_round)
handler.global_voting()
"""
# ================================================================================================================
"""
# 랜덤하게 샤드를 선택하여 각 모델에 대해 이전 모델과 현재 업데이트된 모델에 대하여 투표한 후 FedAvg를 한다.
handler = Voting(current_round) # init 시 voting 시작
handler.model_voter() # 첫번째 voting
# handler.model_voter() # 두번째 voting
# handler.model_voter() # 세번째 voting
# handler.model_voter() # 네번째 voting 총 5개의 shard

# mitigate model update
previous_model = load_model("./model/" + str(current_round - 1) + "/aggregation.pt")
current_model = load_model("./model/" + str(current_round) + "/aggregation.pt")

mitigate_model = mitigate_fed_avg(previous_model, current_model, alpha=GLOBAL_MODEL_ALPHA)
torch.save(mitigate_model.state_dict(), "./model/" + str(current_round) + "/aggregation.pt")

for address in p.SHARD_ADDR_LIST:
    for filename in p.FILE_LIST:
        shard = sender.sendServer(address["ip"], address["port"])
        if filename == ".DS_Store":
            pass
        shard.send_file("model/" + str(current_round) + "/" + filename)
        shard.clientSock.close()
        print("socket closed")

# model = load_model("model/2/aggregation.pt")
# actuals, predictions = test_label_predictions(model)
# accuracy = accuracy_score(actuals, predictions) * 100
# print(accuracy)
"""

# ================================================================================================================
# 랜덤하게 샤드를 선택하여 각 모델에 대해 이전 모델과 현재 업데이트된 모델에 대하여 투표한 후 FedAvg를 한다.
if current_round == 1:
    Logger("server_logs" + str(current_round)).log("=========== Round 1 Model Aggregation ===========")
    print("=========== Round 1 Model Aggregation ===========")

    shard1 = load_model("./model/" + str(current_round) + "/shard1.pt")
    shard2 = load_model("./model/" + str(current_round) + "/shard2.pt")
    # shard3 = load_model("./model/" + str(current_round) + "/shard3.pt")
    # shard4 = load_model("./model/" + str(current_round) + "/shard4.pt")
    # shard5 = load_model("./model/" + str(current_round) + "/shard5.pt")

    print("aggregation model")
    # avg = fed_avg(shard1, shard2, shard3, shard4, shard5)
    avg = fed_avg(shard1, shard2)
    torch.save(avg.state_dict(), "./model/" + str(current_round) + "/aggregation.pt")

else:
    handler = Voting(current_round) # init 시 voting 시작
    handler.model_voter() # 첫번째 voting
    # handler.model_voter() # 두번째 voting
    # handler.model_voter() # 세번째 voting
    # handler.model_voter() # 네번째 voting 총 5개의 shard

    # mitigate model update
    previous_model = load_model("./model/" + str(current_round - 1) + "/aggregation.pt")
    current_model = load_model("./model/" + str(current_round) + "/aggregation.pt")

    mitigate_model = mitigate_fed_avg(previous_model, current_model, alpha=GLOBAL_MODEL_ALPHA)
    torch.save(mitigate_model.state_dict(), "./model/" + str(current_round) + "/aggregation.pt")

for address in p.SHARD_ADDR_LIST:
    for filename in p.FILE_LIST:
        shard = sender.sendServer(address["ip"], address["port"])
        if filename == ".DS_Store":
            pass
        shard.send_file("model/" + str(current_round) + "/" + filename)
        shard.clientSock.close()
        print("socket closed")

# model = load_model("model/2/aggregation.pt")
# actuals, predictions = test_label_predictions(model)
# accuracy = accuracy_score(actuals, predictions) * 100
# print(accuracy)
