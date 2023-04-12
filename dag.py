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
from conf.global_settings import HASH_LENGTH, LOG_DIR, DATA_TYPE, SEARCH_SPACE_SIZE


class DAG:
    def __init__(self, args, global_round):
        self.genesis_tx_hash = token_hex(HASH_LENGTH)
        self.pre_global_model = get_network(args)
        """ 
        if global_round - 1 >= 0:
            print("Load previous global model")
            self.pre_global_model.load_state_dict(torch.load("./" + LOG_DIR + "/" + DATA_TYPE + "/" + args.net + "/global_model/G" + str(global_round - 1) + "/aggregation.pt"), strict=True)

            self.transaction_pool = {self.genesis_tx_hash: TX(self.genesis_tx_hash, timestamp=0, previous_hashes=None, approved_tx=[], tx_owner_id="genesis", payload={"model": self.pre_global_model, "projection": 0})}
        else:
        """
        self.transaction_pool = {self.genesis_tx_hash: TX(self.genesis_tx_hash, timestamp=0, previous_hashes=None, approved_tx=[], tx_owner_id="genesis", payload={"model": self.pre_global_model, "projection": 0})}

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
                self.projection_based_selection(worker, time)
                return list(random.sample(set(list(self.transaction_pool.keys())[-2:]), 2))

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
            print_log(worker.logger, f"TX hash: {tx.tx_hash} | Time: {tx.timestamp:2} | Own: {tx.tx_owner_id:8} | Projection {tx.payload['projection']:.2f}")

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
                    tx_timestamp = tx_info.timestamp
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

        print(top_2_max_accuracy_tx)