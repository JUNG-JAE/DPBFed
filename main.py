# ------------ system library ------------ #
import random
import argparse

# ----------- Learning library ----------- #

# ------------ custom library ------------ #
from worker import Worker
from dag import DAG
from system_utility import set_global_round, set_logger, poisson_distribution, print_log, plot_DAG
from conf.global_settings import NUM_OF_WORKER, NUM_OF_MALICIOUS_WORKER, TIME, LEARNING_EPOCH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    args = parser.parse_args()

    global_round = set_global_round(args)
    logger = set_logger(args, global_round)

    workers = []

    for worker_index in range(NUM_OF_WORKER):
        is_malicious = worker_index in range(NUM_OF_MALICIOUS_WORKER)
        worker = Worker(args, f'worker{worker_index}', global_round, logger, is_malicious)
        workers.append(worker)

    poisson = poisson_distribution()
    dag = DAG(args, global_round)

    print_log(logger, "Total num of transaction: {0:3}".format(sum(poisson)))

    for minute in range(1, TIME + 1):
        print_log(logger, "==================== Time:{0:2} (min) ====================".format(minute))
        num_of_round_transaction = poisson[minute - 1]
        workers_of_round = random.sample(workers, num_of_round_transaction)
        print_log(logger, "Transaction creator worker: {0}".format([worker.worker_id for worker in workers_of_round]))

        for worker in workers_of_round:
            print("<------------ {0} invoke transaction ------------>".format(worker.worker_id))
            if worker.malicious_worker:
                worker_model = worker.model_poisoning_attack()
            else:
                worker_model = worker.train_model(LEARNING_EPOCH)

            projection_value = round(worker.model_projection().item(), 4)

            dag.generate_transactions(timestamp=minute, worker=worker, payload={"model": worker_model, "projection": projection_value})
            print(" ")

        print(" ")

    for worker in workers:
        worker.save_tracker(global_round)
    dag.save_global_model()
    plot_DAG(args, global_round, dag)
