# ----------- System library ----------- #
import numpy as np
from secrets import token_hex
import random
# ---------- Learning library ---------- #
import torch

# ----------- Custom library ----------- #
from learning_utility import get_network
from system_utility import print_log
from transaction import transactionBlock as TX
from conf.global_settings import HASH_LENGTH, LOG_DIR, DATA_TYPE, SEARCH_SPACE_SIZE, TIME


class DAG:
    def __init__(self, args, logger, global_round):
        self.args = args
        self.logger = logger
        self.genesis_tx_hash = token_hex(HASH_LENGTH)
        self.pre_global_model = get_network(self.args)
        self.global_round = global_round

        if self.global_round - 1 > 0:
            print_log(self.logger, "DAG: Load previous global model")
            model_path = f"./{LOG_DIR}/{DATA_TYPE}/{args.net}/global_model/G{self.global_round - 1}/aggregation.pt"
            self.pre_global_model.load_state_dict(torch.load(model_path), strict=True)

        genesis_payload = {"model": self.pre_global_model, "projection": 0}
        genesis_tx = TX(self.genesis_tx_hash, timestamp=0, previous_hashes=None, approved_tx=[], tx_owner_id="genesis", payload=genesis_payload)
        self.transaction_pool = {self.genesis_tx_hash: genesis_tx}

        self.edges = {self.genesis_tx_hash: []}
        self.reverse_edges = {self.genesis_tx_hash: []}
        self.worker_cumulative_weight_dict = {}

    def add_transaction(self, transaction: TX):
        self.transaction_pool[transaction.tx_hash] = transaction
        self.add_edges(transaction)
        # self.update_cumulative_weights(transaction)

    def add_edges(self, transaction: TX):
        approved = transaction.approved_tx
        self.edges[transaction.tx_hash] = approved

        self.reverse_edges.setdefault(approved[0], []).append(transaction.tx_hash)
        self.reverse_edges.setdefault(approved[1], []).append(transaction.tx_hash)

    def find_tips(self, worker, time):
        if len(self.transaction_pool) == 1:
            return [self.genesis_tx_hash, self.genesis_tx_hash]
        else:
            if worker.malicious_worker:
                tips = self.random_tip_selection(worker, time)
                return tips
            else:
                tips = self.random_tip_selection(worker, time)
                # tips = self.projection_based_selection(worker, time)
                return tips

    def update_cumulative_weights(self, current_node):
        stack = [current_node]

        while stack:
            node = stack.pop()

            if node.tx_hash != self.genesis_tx_hash:
                for approved_tx in node.approved_tx:
                    child_node = self.transaction_pool[approved_tx]
                    stack.append(child_node)

            node.cumulative_weight += 1

    def get_previous_hashes(self, tips):
        previous_hashes = [self.transaction_pool[tips[0]].getHash(), self.transaction_pool[tips[1]].getHash()]

        return previous_hashes

    def generate_transactions(self, timestamp=0, worker=None, payload=None):
        tips = self.find_tips(worker, time=timestamp)
        previous_hashes = self.get_previous_hashes(tips)
        transaction = TX(tx_hash=token_hex(HASH_LENGTH), timestamp=timestamp, previous_hashes=previous_hashes, approved_tx=tips, tx_owner_id=worker.worker_id, payload=payload)
        self.add_transaction(transaction)

    def random_tip_selection(self, worker, time):
        print_log(worker.logger, "----------- Search space -----------")
        search_space_time = np.clip(range((time - SEARCH_SPACE_SIZE) + 1, time + 1), 0, None)
        searched_tx = [tx for key_hash, tx in self.transaction_pool.items() if tx.timestamp in search_space_time]

        for tx in searched_tx:
            print_log(worker.logger, f"TX hash: {tx.tx_hash} | Time: {tx.timestamp:2} | Own: {tx.tx_owner_id:8} | Projection {tx.payload['projection']:.2f} | Accuracy 0")

        randomly_selected_tips = random.sample(searched_tx, 2)

        return [randomly_selected_tips[0].tx_hash, randomly_selected_tips[1].tx_hash]

    def dfs_topological_sort_from_node(self, start_node):
        visited = set()
        stack = [start_node]
        traversal_order = []

        while stack:
            current_node = stack[-1]

            if current_node not in visited:
                visited.add(current_node)
                for neighbor in self.edges[current_node]:
                    if neighbor not in visited:
                        stack.append(neighbor)
            else:
                stack.pop()
                traversal_order.append(current_node)

        return traversal_order[::-1]

    def projection_based_selection(self, worker, time):
        print_log(worker.logger, "----------- Search space -----------")
        search_space_time = np.clip(range((time - SEARCH_SPACE_SIZE) + 1, time + 1), 0, None)
        searched_tx = [tx for key_hash, tx in self.transaction_pool.items() if tx.timestamp in search_space_time]

        for tx in searched_tx:
            traversal_order = self.dfs_topological_sort_from_node(tx.tx_hash)

            for tx_hash in traversal_order:
                tx_info = self.transaction_pool[tx_hash]
                owner_id = tx_info.tx_owner_id

                if owner_id != "genesis":
                    tx_timestamp = tx_info.timestamp + (TIME * self.global_round)
                    tx_projection = tx_info.payload['projection']
                    worker.peer_tracker.setdefault(owner_id, {})[tx_timestamp] = tx_projection

        worker.update_peer_change_rate()
        worker.predict_malicious_worker()

        print("Black list: {0}".format(worker.peer_black_list))
        filtered_searched_tx = [tx for tx in searched_tx if tx.tx_owner_id not in worker.peer_black_list]

        tx_accuracy = {}
        for tx in filtered_searched_tx:
            accuracy = worker.evaluate_model(tx.payload['model'])
            print_log(worker.logger, f"TX hash: {tx.tx_hash} | Time: {tx.timestamp:2} | Own: {tx.tx_owner_id:8} | Projection {tx.payload['projection']:.2f} | Accuracy {accuracy:.2f}")
            tx_accuracy[tx.tx_hash] = accuracy

        sorted_tx = sorted(tx_accuracy.items(), key=lambda item: item[1], reverse=True)
        top_2_max_accuracy_tx = sorted_tx[:2]

        print("{0} -> {1} ({2})".format(worker.worker_id, self.transaction_pool[top_2_max_accuracy_tx[0][0]].tx_owner_id, top_2_max_accuracy_tx[0][0]))
        print("{0} -> {1} ({2})".format(worker.worker_id, self.transaction_pool[top_2_max_accuracy_tx[1][0]].tx_owner_id, top_2_max_accuracy_tx[1][0]))

        return [top_2_max_accuracy_tx[0][0], top_2_max_accuracy_tx[1][0]]
    """ 
    def save_global_model(self):
        valid_tx = [tx for tx in self.transaction_pool.values() if tx.timestamp >= (TIME - SEARCH_SPACE_SIZE)]
        random.shuffle(valid_tx)

        maximum_transaction_hash = None
        maximum_model_site_num = 0

        for tx in valid_tx:
            if tx.tx_hash in self.reverse_edges:
                site_num = len(self.reverse_edges[tx.tx_hash])
            else:
                site_num = 0
            if site_num > maximum_model_site_num:
                maximum_model_site_num = site_num
                maximum_transaction_hash = tx.tx_hash
            print("{0}, time: {1}, site {2}".format(tx.tx_owner_id, tx.timestamp, site_num))

        print("maximum model({0}): {1}, site: {2}".format(maximum_transaction_hash, self.transaction_pool[maximum_transaction_hash].tx_owner_id, maximum_model_site_num))

        shard_model = self.transaction_pool[maximum_transaction_hash].payload['model']

        torch.save(shard_model.state_dict(), LOG_DIR + "/" + DATA_TYPE + "/" + self.args.net + "/global_model/G" + str(self.global_round) + "/aggregation.pt")
    """
    def save_global_model(self):
        valid_tx = [tx for tx in self.transaction_pool.values() if tx.timestamp >= (TIME - SEARCH_SPACE_SIZE)]
        # random.shuffle(valid_tx)

        maximum_transaction = max(valid_tx, key=lambda tx: len(self.reverse_edges.get(tx.tx_hash, [])))
        maximum_model_site_num = len(self.reverse_edges.get(maximum_transaction.tx_hash, []))

        for tx in valid_tx:
            site_num = len(self.reverse_edges.get(tx.tx_hash, []))
            print(f"{tx.tx_owner_id}, time: {tx.timestamp}, site {site_num}")

        print(f"maximum model({maximum_transaction.tx_hash}): {maximum_transaction.tx_owner_id}, site: {maximum_model_site_num}")

        shard_model = maximum_transaction.payload['model']
        torch.save(shard_model.state_dict(), f"{LOG_DIR}/{DATA_TYPE}/{self.args.net}/global_model/G{self.global_round}/aggregation.pt")
