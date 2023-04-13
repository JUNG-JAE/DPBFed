# ----------- Shard parameters ----------- #
SHARD_ID = "shard1"
NUM_OF_WORKER = 10
NUM_OF_MALICIOUS_WORKER = 5
SEARCH_SPACE_SIZE = 10

# ----------- Worker parameters ----------- #
DATA_TYPE = "FMNIST"
CHANNEL_SIZE = 1 if DATA_TYPE in ["MNIST", "FMNIST"] else 3
LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
BATCH_SIZE = 64
LEARNING_RATE = 0.001
LEARNING_EPOCH = 1
RANDOM_TRAINING = False

# ----------- Attack parameters ----------- #
POISONED_ATTACK_RATE = 1

GAUSSIAN_MEAN = 0
GAUSSIAN_STD = 1

# ----------- Defense parameters ----------- #
CDF_LOWER_BOUND = 0.85

# ----------- System parameters ----------- #
HASH_LENGTH = 8
TIME = 20
TRANSACTION_PER_MINUTE = 0.13 * NUM_OF_WORKER
LOG_DIR = "./runs"






