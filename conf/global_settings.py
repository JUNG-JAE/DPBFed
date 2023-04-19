# ----------- Shard parameters ----------- #
SHARD_ID = "shard1"
NUM_OF_WORKER = 10
NUM_OF_MALICIOUS_WORKER = 4
SEARCH_SPACE_SIZE = 10

# ----------- Worker parameters ----------- #
DATA_TYPE = "FMNIST"
CHANNEL_SIZE = 1 if DATA_TYPE in ["MNIST", "FMNIST"] else 3

if DATA_TYPE == "MNIST":
    LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
elif DATA_TYPE == "FMNIST":
    LABELS = ['Bag', 'Boot', 'Coat', 'Dress', 'Pullover', 'Sandal', 'Shirt', 'Sneaker', 'Top', 'Trouser']
elif DATA_TYPE == "CIFAR10":
    LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

BATCH_SIZE = 256
LEARNING_RATE = 0.001
LEARNING_EPOCH = 3
RANDOM_TRAINING = False

# ----------- Attack parameters ----------- #
POISONED_ATTACK_RATE = 1

GAUSSIAN_MEAN = 0
GAUSSIAN_STD = 1

# ----------- Defense parameters ----------- #
CDF_LOWER_BOUND = 0.8

# ----------- System parameters ----------- #
HASH_LENGTH = 8
TIME = 30
TRANSACTION_PER_MINUTE = 0.13 * NUM_OF_WORKER
LOG_DIR = "./runs"

# ----------- Server parameters ----------- #
NUM_OF_SHARD = 2

SERVER_IP = '127.0.0.1'
SERVER_PORT = 9025

SHARD_IP = '127.0.0.1'
SHARD_PORT = 9001






