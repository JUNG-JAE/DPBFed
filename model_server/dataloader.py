import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io
from glob import glob
import torchvision.transforms as transforms

from parameter import labels, MINI_BATCH

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def get_label(data_path_list):
    label_list = []
    for path in data_path_list:
        # 뒤에서 두번째가 class다.
        label_list.append(path.split('/')[-2])
    return label_list


class MyData(Dataset):
    def __init__(self, data_path_list, classes, transform=None):
        self.path_list = data_path_list  # cifar10 이미지 경로
        self.label = get_label(data_path_list)  # 클래스 이름
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.path_list)  # 전체 데이터 셋의 길이 반환

    def __getitem__(self, idx):
        if torch.is_tensor(idx):  # input값이 텐서형태이기 때문에 이를 리스트로 변환해준다.
            idx = idx.tolist()
        image = io.imread(self.path_list[idx])  # index에 해당하는 이미지를 가져온다.
        if self.transform is not None:
            image = self.transform(image)
        return image, self.classes.index(self.label[idx])


def get_dataloader(type, worker=None):
    # mnist 데이터 가져오기 =================================================================
    if type == "all":
        DATA_PATH_TESTING_LIST = glob('./data/test/*/*.png')
    elif type == "shard1":
        # 0, 1
        DATA_PATH_TESTING_LIST = glob('./data/shard1/worker'+str(worker)+'/test/*/*.png')
    elif type == "shard2":
        # 2, 3
        DATA_PATH_TESTING_LIST = glob('./data/shard2/worker' + str(worker) + '/test/*/*.png')
    elif type == "shard3":
        # 4, 5
        DATA_PATH_TESTING_LIST = glob('./data/shard3/worker' + str(worker) + '/test/*/*.png')
    elif type == "shard4":
        # 6, 7
        DATA_PATH_TESTING_LIST = glob('./data/shard4/worker' + str(worker) + '/test/*/*.png')
    elif type == "shard5":
        # 8, 9
        DATA_PATH_TESTING_LIST = glob('./data/shard5/worker' + str(worker) + '/test/*/*.png')
    # ===================================================================================

    testloader = torch.utils.data.DataLoader(
        MyData(
            DATA_PATH_TESTING_LIST,
            labels,
            # transform=transform  # for cifar-10
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]) # for mnist dataset
        ),
        batch_size=MINI_BATCH,
        shuffle=True
    )

    return testloader

