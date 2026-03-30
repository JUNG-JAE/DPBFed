import shutil
import os
import torch
import warnings
from model import CNN
from round_checker import model_loader
from parameter import SHARD_ID
warnings.filterwarnings('ignore')

# BLOCK_NUM = input("Input Block Number: ")
BLOCK_NUM = max(model_loader())

# Copy the experiment data
if SHARD_ID not in os.listdir("./../"):
    print("{0} Directory produced !".format(SHARD_ID))
    os.mkdir("./../" + SHARD_ID)

# ./../s1/1 -> ./../shard1/1
if str(BLOCK_NUM) not in os.listdir("./../" + SHARD_ID + "/"):
    print("Block {0} Directory produced !".format(BLOCK_NUM))
    os.mkdir("./../" + SHARD_ID + "/" + str(BLOCK_NUM))

shutil.copytree("./acc_graph_data", "./../" + SHARD_ID + "/" + str(BLOCK_NUM) + "/acc_graph_data/")
shutil.copytree("./graph", "./../" + SHARD_ID + "/" + str(BLOCK_NUM) + "/graph/")
shutil.copytree("./logs", "./../" + SHARD_ID + "/" + str(BLOCK_NUM) + "/logs/")
shutil.copytree("./migrate", "./../" + SHARD_ID + "/" + str(BLOCK_NUM) + "/migrate/")
shutil.copytree("./model/" + str(BLOCK_NUM) + "/", "./../" + SHARD_ID + "/" + str(BLOCK_NUM) + "/model/")

shutil.rmtree('./logs', ignore_errors=True)
shutil.rmtree('./migrate', ignore_errors=True)
shutil.rmtree('./graph', ignore_errors=True)
shutil.rmtree('./acc_graph_data', ignore_errors=True)

os.mkdir('./logs')
os.mkdir('./migrate')
os.mkdir('./graph')
os.mkdir('./acc_graph_data')






