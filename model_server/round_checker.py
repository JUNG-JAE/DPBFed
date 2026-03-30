import os

SAVE_MODEL_PATH = "./model/"

def model_loader():
    rounds = os.listdir(SAVE_MODEL_PATH)

    for i, round in enumerate(rounds):
        try:
            rounds[i] = int(round)
            rounds = list(map(int, rounds))
        except Exception as ex:
            del rounds[i]
    return rounds

def current_round_checker():
    check_first = model_loader()

    if len(check_first) == 0:
        os.mkdir(SAVE_MODEL_PATH + "1")

        print("current round: {0}".format(1))

        return 1
    else:
        last_round = model_loader()
        os.mkdir(SAVE_MODEL_PATH + str(int(max(last_round)) + 1))

        print("current round: {0}".format(str(int(max(last_round)) + 1)))

        return int(max(last_round)) + 1

# round = current_round_checker()
