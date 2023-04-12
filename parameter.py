"""
# main
WORKER_NUM = 3  # total number of  local worker in shard
TOTAL_ROUND = 3  # total round (number of for loop)
TRAINING_EPOCH = 1  # training epoch for each worker per round

# for random training epoch each node
RANDOM_TRAINING_EPOCH = False  # setup random training epoch
MIN_EPOCH = 1 # minimum training epoch
MAX_EPOCH = 5 # maximum training epoch

# possion parameter
LAM = 8
SIZE = TOTAL_ROUND

# tip selection algorithmn
TIP_SELECT_ALGO = 'high_accuracy'
LEARNING_MEASURE = "f1 score"
SIMILARITY_WEIGHT = 0.05
MULTIPLICITY_WEIGHT = 0.01
MODE = "MultiObject"

# file save path
SAVE_SHARD_MODEL_PATH = './model/'
SAVE_MIGRATION_INFO_PATH = './migrate/shard1.txt'

# dag PoW difficulity
DIFFICULTY = 0

# setup for worker
MINI_BATCH_SIZE = 64
LEARNING_RATE = 0.0001

# socket connection
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 9025

# migration server socket info
MIGRATION_SERVER_HOST = '127.0.0.1'
MIGRATION_SERVER_PORT = 9045

# aggregation server socket info
SHARD_HOST = '127.0.0.1'
SHARD_PORT = 9010

# shard ID
SHARD_ID = "shard1"

# total number of shard
SHARD_NUM = 5

# all shard list
shard_list = {"shard1": [], "shard2": [], "shard3": [], "shard4": [], "shard5": []}

# data set labels
label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # Mnist

# shard's sub label
sub_label = ['0', '1']

ATTACK_TYPE = "MODEL_POISONING"

# model poisoning attack
POISON_WORKER = []
GAUSSIAN_MEAN = 0
GAUSSIAN_SIGMA = 2
COWORK = False
MALICIOUS_LEADER = []

# FGSM attack
EPSILON = 0.9

# PGD Attack
ALPHA = 0.5
STEP = 40

# Data poisoning attack (Gaussian Noise Attack)
NOISE_MEAN = 10
NOISE_STD = 1

# # total number of upload models list
# UPLOAD_MODEL_NUM = 3
# MULTI_UPLOAD = False

QUANTIZATION = True
"""
