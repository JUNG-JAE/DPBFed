from collections import OrderedDict
import time
import hashlib
from util import Logger

class transaction_block:
    def __init__(self, sender_public_key, recipient_public_key, payload, signature, tx_id, approved_tx, nonce, previous_hashes, tx_worker_id):
        self.tx_id = tx_id
        self.own_weight = 1
        self.cumulative_weight = 0
        self.nonce = nonce
        self.previous_hashes = previous_hashes
        self.approved_tx = approved_tx,
        self.timestamp = time.time(),
        self.signature = signature
        self.recipient_public_key = recipient_public_key
        self.sender_public_key = sender_public_key
        self.payload = payload

        self.tx_worker_id = tx_worker_id

    def get_hash(self):
        transaction_dict = OrderedDict({
            'sender_public_key': self.sender_public_key,
            'recipient_public_key': self.recipient_public_key,
            'payload': self.payload
        })
        return hash_data(transaction_dict)

    def show(self):
        Logger('transaction').log("Transaction ID = {0}".format(self.tx_id))
        Logger('transaction').log("Time Stamp = {0}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp[0]))))
        Logger('transaction').log("Transaction own Worker ID : {0}".format(self.tx_worker_id))
        Logger('transaction').log("Cumulative Weight = {0}".format(self.cumulative_weight))
        Logger('transaction').log(' ')

    def get_payload(self):
        return self.payload

def hash_data(data):
    data = str(data).encode('utf-8')
    h = hashlib.new('sha256')
    h.update(data)
    return h.hexdigest()