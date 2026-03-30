import os
from parameter import SAVE_SHARD_MODEL_PATH

def model_loader():
    rounds = os.listdir(SAVE_SHARD_MODEL_PATH)

    for i, round in enumerate(rounds):
        try:
            rounds[i] = int(round)
            rounds = list(map(int, rounds))
        except Exception:
            del rounds[i]
    return rounds

def current_round_checker():
    check_first = model_loader()

    if len(check_first) == 0:
        os.mkdir(SAVE_SHARD_MODEL_PATH + "1")
        print("############## Global Current Round: 1 ##############")

        return 1
    else:
        last_round = model_loader()
        os.mkdir(SAVE_SHARD_MODEL_PATH + str(int(max(last_round)) + 1))
        print("############## Global Current Round: {0} ##############".format(str(int(max(last_round)) + 1)))

        return int(max(last_round)) + 1

def worker_current_round_chcker():
    check_first = model_loader()

    if len(check_first) == 0:
        print("Current Round: 1")

        return 1
    else:
        last_round = model_loader()
        print("Current Round: {0}".format(str(int(max(last_round)))))

        return int(max(last_round))