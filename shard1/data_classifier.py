import shutil
import os
import parameter as p


worker_train_dict = {}
worker_test_dict = {}

"""
train_data_path = {'0': "./mnist_png/train/0",
                     '1': "./mnist_png/train/1",
                     '2': "./mnist_png/train/2",
                     '3': "./mnist_png/train/3",
                     '4': "./mnist_png/train/4",
                     '5': "./mnist_png/train/5",
                     '6': "./mnist_png/train/6",
                     '7': "./mnist_png/train/7",
                     '8': "./mnist_png/train/8",
                     '9': "./mnist_png/train/9"}

test_data_path = {'0': "./mnist_png/test/0",
                    '1': "./mnist_png/test/1",
                    '2': "./mnist_png/test/2",
                    '3': "./mnist_png/test/3",
                    '4': "./mnist_png/test/4",
                    '5': "./mnist_png/test/5",
                    '6': "./mnist_png/test/6",
                    '7': "./mnist_png/test/7",
                    '8': "./mnist_png/test/8",
                    '9': "./mnist_png/test/9"}
"""

train_data_path = {'0': "./fashion_mnist/train/Bag",
                     '1': "./fashion_mnist/train/Boot",
                     '2': "./fashion_mnist/train/Coat",
                     '3': "./fashion_mnist/train/Dress",
                     '4': "./fashion_mnist/train/Pullover",
                     '5': "./fashion_mnist/train/Sandal",
                     '6': "./fashion_mnist/train/Shirt",
                     '7': "./fashion_mnist/train/Sneaker",
                     '8': "./fashion_mnist/train/Top",
                     '9': "./fashion_mnist/train/Trouser"}

test_data_path = {'0': "./fashion_mnist/test/Bag",
                     '1': "./fashion_mnist/test/Boot",
                     '2': "./fashion_mnist/test/Coat",
                     '3': "./fashion_mnist/test/Dress",
                     '4': "./fashion_mnist/test/Pullover",
                     '5': "./fashion_mnist/test/Sandal",
                     '6': "./fashion_mnist/test/Shirt",
                     '7': "./fashion_mnist/test/Sneaker",
                     '8': "./fashion_mnist/test/Top",
                     '9': "./fashion_mnist/test/Trouser"}

def create_folder(worker_directory, class_name, d_type):
    try:
        if not os.path.exists(worker_directory + "/" + d_type + "/" + class_name):
            os.makedirs(worker_directory + "/" + d_type + "/" + class_name)
    except OSError:
        print("Error: Creating directory. " + worker_directory + "/" + class_name)


def data_copy(worker_name, data_type):
    if data_type == "train":
        for class_num in range(0, 10):
            image_index_list = list(worker_train_dict[worker_name][class_num].values())[0]

            for index in image_index_list:
                shutil.copy2(train_data_path[str(class_num)] + "/" + str(index) + ".png","./data/" + str(worker_name) + "/" + data_type + "/" + "/" + str(class_num) + "/" + str(index) + ".png")
    elif data_type == "test":
        for class_num in range(0, 10):
            image_index_list = list(worker_test_dict[worker_name][class_num].values())[0]

            for index in image_index_list:
                shutil.copy2(test_data_path[str(class_num)] + "/" + str(index) + ".png", "./data/" + str(worker_name) + "/" + data_type + "/" + "/" + str(class_num) + "/" + str(index) + ".png")

def migrate_worker():
    migration_data_path = "./migrate/migration_info.txt"
    data = open(migration_data_path, 'r')
    data = eval(data.read())

    train_index = data["train"][p.SHARD_ID]
    test_index = data["test"][p.SHARD_ID]

    worker_num = len(train_index)

    shutil.rmtree("./data")
    print(" ")
    print("* * * * * * * * * * * * * * * * * *")
    print("*      previous data removed      *")

    for i in range(0, worker_num):
        worker_train_dict.update({'worker' + str(i): []})
        worker_test_dict.update({'worker' + str(i): []})

    for i in range(0, worker_num):
        for j in range(0, 10):
            worker_train_dict['worker' + str(i)].append(train_index[i][j])
            worker_test_dict['worker' + str(i)].append(test_index[i][j])

    for worker_index in range(0, worker_num):
        for class_index in range(0, 10):
            create_folder("./data/worker" + str(worker_index), str(class_index), "train")

    for worker_index in range(0, worker_num):
        for class_index in range(0, 10):
            create_folder("./data/worker" + str(worker_index), str(class_index), "test")

    for worker_index in range(0, worker_num):
        data_copy("worker" + str(worker_index), "train")

    for worker_index in range(0, worker_num):
        data_copy("worker" + str(worker_index), "test")

    print("*   shard next worker number: {0}   *".format(worker_num))
    print("* * * * * * * * * * * * * * * * * *")
    print(" ")

migrate_worker()