import torch
from torch.utils.data import Dataset
from skimage import io
from glob import glob

import torchvision.transforms as transforms

import parameter as p

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def get_label(data_path_list):
    label_list = []
    for path in data_path_list:
        # 뒤에서 두번째가 class다.
        label_list.append(path.split('/')[-2])
    return label_list


class MyCifarSet(Dataset):
    # data_path_list - 이미지 path 전체 리스트
    # label - 이미지 ground truth
    def __init__(self, data_path_list, classes, transform=None):
        self.path_list = data_path_list  # cifar10 이미지 경로
        self.label = get_label(data_path_list)  # 클래스 이름
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.path_list)  # 전체 데이터 셋의 길이 반환

    # 학습에 사용할 이미지 변환, path_list에서index에 해당하는 이미지를 읽는다.
    def __getitem__(self, idx):
        if torch.is_tensor(idx):  # input값이 텐서형태이기 때문에 이를 리스트로 변환해준다.
            idx = idx.tolist()
        image = io.imread(self.path_list[idx])  # index에 해당하는 이미지를 가져온다.
        if self.transform is not None:
            image = self.transform(image)
        return image, self.classes.index(self.label[idx])


def set_dataloader(worker_id):
    if worker_id == 'genesis_worker':
        DATA_PATH_TRAINING_LIST = glob('./data/' + 'worker0' + '/train/*/*.png')
        DATA_PATH_TESTING_LIST = glob('./data/' + 'worker0' + '/test/*/*.png')
        print("genesis worker data loaded!")
    else:
        DATA_PATH_TRAINING_LIST = glob('./data/' + str(worker_id) + '/train/*/*.png')
        DATA_PATH_TESTING_LIST = glob('./data/' + str(worker_id) + '/test/*/*.png')
        print("{0} data loaded!".format(worker_id))
    # DATA_PATH_TRAINING_LIST = glob('./data/mnist_png/train/*/*.png')
    # DATA_PATH_TESTING_LIST = glob('./data/mnist_png/test/*/*.png')

    trainloader = torch.utils.data.DataLoader(
        MyCifarSet(
            DATA_PATH_TRAINING_LIST,
            p.label,
            # transform=transform
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])  # for mnist dataset
        ),
        batch_size=p.MINI_BATCH_SIZE,
        shuffle=True
    )

    testloader = torch.utils.data.DataLoader(
        MyCifarSet(
            DATA_PATH_TESTING_LIST,
            p.label,
            # transform=transform
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])  # for mnist dataset
        ),
        batch_size=p.MINI_BATCH_SIZE,
        shuffle=True
    )

    return trainloader, testloader


def origin_test_loader():
    DATA_PATH_TESTING_LIST = glob('./data/mnist_png/test/*/*.png')

    testloader = torch.utils.data.DataLoader(
        MyCifarSet(
            DATA_PATH_TESTING_LIST,
            p.label,
            # transform=transform
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])  # for mnist dataset
        ),
        batch_size=p.MINI_BATCH_SIZE,
        shuffle=True
    )

    return testloader

