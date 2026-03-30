import warnings
import torch.nn.init
import torch.nn.functional as F
from torch import nn
from functools import reduce
import os

from model import CNN
import dataloader
from util import Logger, gaussian_distribution
import parameter as p
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torchattacks
from util import quantize

warnings.filterwarnings(action='ignore')

# GPU가 없을 경우, CPU를 사용한다.
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

GAUSSIAN_RANGE = np.arange(-1, 1, 0.00001)

class Worker:
    def __init__(self, worker_id, current_round, poisoned=False):
        self.worker_id = worker_id
        print("{0} setup".format(self.worker_id))
        self.round = 0
        self.delay = False
        self.poisoned = poisoned
        self.approve_list = {}
        self.total_training_epoch = 0
        self.selected_transaction = 0

        self.data_loader, self.test_loader = dataloader.set_dataloader(worker_id)
        self.origin_test_loader = dataloader.origin_test_loader()

        self.model = CNN().to(device)
        self.trained_model = CNN().to(device)

        if current_round-1 > 0:
            self.model.load_state_dict(torch.load("./model/" + str(current_round - 1) + "/aggregation.pt"), strict=False)
            self.trained_model.load_state_dict(torch.load("./model/" + str(current_round - 1) + "/aggregation.pt"), strict=False)
            print("[{0}]: global model inital".format(self.worker_id))

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=p.LEARNING_RATE)
        self.total_batch = len(self.data_loader)
        # self.time_length, _ = list(np.histogram(np.random.poisson(p.LAM, p.SIZE), bins=np.array(p.SIZE))) # poisson 분포
        # print("WorkerId: {0}, time_length: {1}".format(self.worker_id, self.time_length))


    def loacl_learning(self, training_epochs=0):
        Logger(str(self.worker_id)).log('Input training epochs: {0}'.format(training_epochs))
        Logger(str(self.worker_id)).log('Total training epochs: {0}'.format(self.total_training_epoch))

        for epoch in range(training_epochs):
            avg_cost = 0

            for data, target in self.data_loader:  # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
                data, target = data.to(device), target.to(device)
                # target = target.to(device)

                self.optimizer.zero_grad()
                hypothesis = self.model(data)
                cost = self.criterion(hypothesis, target)
                cost.backward()
                self.optimizer.step()

                avg_cost += cost / self.total_batch
            Logger(str(self.worker_id)).log('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

        if p.QUANTIZATION:
            self.model.layer1[0].weight.data = quantize(self.model.layer1[0].weight.data)
            self.model.layer1[0].bias.data = quantize(self.model.layer1[0].bias.data)

            self.model.layer2[0].weight.data = quantize(self.model.layer2[0].weight.data)
            self.model.layer2[0].bias.data = quantize(self.model.layer2[0].bias.data)

            self.model.layer3[0].weight.data = quantize(self.model.layer3[0].weight.data)
            self.model.layer3[0].bias.data = quantize(self.model.layer3[0].bias.data)

            self.model.fc1.weight.data = quantize(self.model.fc1.weight.data)
            self.model.fc1.bias.data = quantize(self.model.fc1.bias.data)

            self.model.fc2.weight.data = quantize(self.model.fc2.weight.data)
            self.model.fc2.bias.data = quantize(self.model.fc2.bias.data)


    def FGSM_attack(self, training_epochs=0):
        Logger(str(self.worker_id)).log('Input training epochs: {0}'.format(training_epochs))
        Logger(str(self.worker_id)).log('Total training epochs: {0}'.format(self.total_training_epoch))

        for epoch in range(training_epochs):
            avg_cost = 0

            fgsm = torchattacks.FGSM(self.trained_model, eps=p.EPSILON)

            for data, target in self.data_loader:
                data, target = data.to(device), target.to(device)
                data = fgsm(data, target)

                self.optimizer.zero_grad()
                hypothesis = self.model(data)
                cost = self.criterion(hypothesis, target)
                cost.backward()
                self.optimizer.step()

                avg_cost += cost / self.total_batch
            Logger(str(self.worker_id)).log('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

    # 디렉토리에서 label을 flipping 한다. 따라서, 일반 학습과 동일하다.
    def label_flipping_attack(self, training_epochs=0):
        Logger(str(self.worker_id)).log('Input training epochs: {0}'.format(training_epochs))
        Logger(str(self.worker_id)).log('Total training epochs: {0}'.format(self.total_training_epoch))

        for epoch in range(training_epochs):
            avg_cost = 0

            for data, target in self.data_loader:  # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
                data, target = data.to(device), target.to(device)

                self.optimizer.zero_grad()
                hypothesis = self.model(data)
                cost = self.criterion(hypothesis, target)
                cost.backward()
                self.optimizer.step()

                avg_cost += cost / self.total_batch
            Logger(str(self.worker_id)).log('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))


    def data_noise_attack(self, training_epochs=0):
        Logger(str(self.worker_id)).log('Input training epochs: {0}'.format(training_epochs))
        Logger(str(self.worker_id)).log('Total training epochs: {0}'.format(self.total_training_epoch))

        for epoch in range(training_epochs):
            avg_cost = 0

            for data, target in self.data_loader:  # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
                data, target = data.to(device), target.to(device)
                data = add_noise(data)

                self.optimizer.zero_grad()
                hypothesis = self.model(data)
                cost = self.criterion(hypothesis, target)
                cost.backward()
                self.optimizer.step()

                avg_cost += cost / self.total_batch
            Logger(str(self.worker_id)).log('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))


    def test_label_predictions(self, model, mode="normal", print_logs=False):
        model.eval()
        actuals = []
        predictions = []

        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        with torch.no_grad():
            if mode == "normal":
                test_data_loader = self.test_loader
            elif mode == "origin":
                test_data_loader = self.origin_test_loader

            for data, target in test_data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                prediction = output.argmax(dim=1, keepdim=True)

                actuals.extend(target.view_as(prediction))
                predictions.extend(prediction)

                if print_logs and mode == "origin":
                    _, predicted = torch.max(output, 1)
                    c = (predicted == target).squeeze()
                    for i in range(4):
                        label = target[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1

            if print_logs and mode == "origin":
                for i in range(10):
                    label_accuracy = round(100 * class_correct[i] / class_total[i], 2)
                    print('accuracy of {0} : {1:.2f}'.format(p.label[i], label_accuracy))

        return [i.item() for i in actuals], [i.item() for i in predictions]


    def evaluation(self, model, print_logs=False, mode=p.LEARNING_MEASURE):
        if mode == "accuracy":
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in self.test_loader:
                    output = model(data)
                    test_loss += self.criterion(output, target).item()  # sum up batch loss
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

                if print_logs:
                    print("accuracy: {0}".format(100 * correct / len(self.test_loader.dataset)))
                return 100 * correct / len(self.test_loader.dataset)

        elif mode == "f1 score":
            actuals, predictions = self.test_label_predictions(model, "normal", print_logs)

            f1 = f1_score(actuals, predictions, average='weighted') * 100
            accuracy = accuracy_score(actuals, predictions) * 100

            if print_logs:
                print("F1 Score: {0} | Accuracy: {1}".format(f1, accuracy))

            return f1

        elif mode == "origin":
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in self.origin_test_loader:
                    output = model(data)
                    test_loss += self.criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

                if print_logs:
                    print("accuracy: {0}".format(100 * correct / len(self.origin_test_loader.dataset)))
                return 100 * correct / len(self.origin_test_loader.dataset)


    def model_aggregation(self, model1, model2):
        self.model.layer1[0].weight.data = (model1.layer1[0].weight.data + model2.layer1[0].weight.data + self.model.layer1[0].weight.data) / 3.0
        self.model.layer1[0].bias.data = (model1.layer1[0].bias.data + model2.layer1[0].bias.data + self.model.layer1[0].bias.data) / 3.0

        self.model.layer2[0].weight.data = (model1.layer2[0].weight.data + model2.layer2[0].weight.data + self.model.layer2[0].weight.data) / 3.0
        self.model.layer2[0].bias.data = (model1.layer2[0].bias.data + model2.layer2[0].bias.data + self.model.layer2[0].bias.data) / 3.0

        self.model.layer3[0].weight.data = (model1.layer3[0].weight.data + model2.layer3[0].weight.data + self.model.layer3[0].weight.data) / 3.0
        self.model.layer3[0].bias.data = (model1.layer3[0].bias.data + model2.layer3[0].bias.data + self.model.layer3[0].bias.data) / 3.0

        self.model.fc1.weight.data = (model1.fc1.weight.data + model2.fc1.weight.data + self.model.fc1.weight.data) / 3.0
        self.model.fc1.bias.data = (model1.fc1.bias.data + model2.fc1.bias.data + self.model.fc1.bias.data) / 3.0

        self.model.fc2.weight.data = (model1.fc2.weight.data + model2.fc2.weight.data + self.model.fc2.weight.data) / 3.0
        self.model.fc2.bias.data = (model1.fc2.bias.data + model2.fc2.bias.data + self.model.fc2.bias.data) / 3.0


    def weight_poison_attack(self):
        self.model.layer1[0].weight.data += noise_constructor(self.model.layer1[0].weight.size())
        self.model.layer1[0].bias.data += noise_constructor(self.model.layer1[0].bias.size())

        self.model.layer2[0].weight.data += noise_constructor(self.model.layer2[0].weight.size())
        self.model.layer2[0].bias.data += noise_constructor(self.model.layer2[0].bias.size())

        self.model.layer3[0].weight.data += noise_constructor(self.model.layer3[0].weight.size())
        self.model.layer3[0].bias.data += noise_constructor(self.model.layer3[0].bias.size())

        self.model.fc1.weight.data += noise_constructor(self.model.fc1.weight.size())
        self.model.fc1.bias.data += noise_constructor(self.model.fc1.bias.size())

        self.model.fc2.weight.data += noise_constructor(self.model.fc2.weight.size())
        self.model.fc2.bias.data += noise_constructor(self.model.fc2.bias.size())

# for model poisoning attack
def noise_constructor(dim):
    tensor_length = reduce(lambda x, y: x * y, dim)
    gaussian = gaussian_distribution(GAUSSIAN_RANGE, p.GAUSSIAN_MEAN, p.GAUSSIAN_SIGMA)
    noise_vector = np.random.choice(gaussian, tensor_length, replace=True)

    noise_dim_split = noise_vector.reshape(dim)
    noise_tensor = torch.Tensor(noise_dim_split)

    return noise_tensor


def FGSM(data, epsilon, data_grad):
    pert_out = data + epsilon * data_grad.sign()
    pert_out = torch.clamp(pert_out, 0, 1)

    return pert_out

# for data poisoning attack
def add_noise(_data):
    # gaussian noise
    # randan: N(0,1)-평균:0, 표준편차1 인 가우시안 분포를 따르는 값을 생성한다.
    # noise = sigma * torch.randn(100, 1, 28, 28) # MNIST image set
    noise = torch.normal(mean=p.NOISE_MEAN, std=p.NOISE_STD, size=(_data.size()))
    contaminate_data = _data + noise

    return contaminate_data



