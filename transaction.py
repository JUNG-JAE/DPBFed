

class TransactionBlock:
    def __init__(self, tx_hash, timestamp, previous_hashes, approved_tx, tx_owner_id, payload):
        self.tx_hash = tx_hash
        self.own_weight = 1
        self.cumulative_weight = 0
        self.timestamp = timestamp
        self.previous_hashes = previous_hashes
        self.approved_tx = approved_tx
        self.tx_owner_id = tx_owner_id
        self.payload = payload


    def getHash(self):
        return self.tx_hash


    def getPayload(self):
        return self.payload

