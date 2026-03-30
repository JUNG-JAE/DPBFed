from model import CNN
import torch
import os
from dataloader import get_dataloader
from parameter import labels


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

def load_model(load_path):
    model = CNN().to(device)
    model.load_state_dict(torch.load(load_path), strict=False)
    return model


def load_worker(shard):
    workers = os.listdir("./data/" + shard + "/")
    for worker in workers:
        if "worker" not in worker:
            workers.remove(worker)

    return workers


def model_fraction(model, numerator, denominator):
    model.layer1[0].weight.data = numerator * model.layer1[0].weight.data / denominator
    model.layer1[0].bias.data = numerator * model.layer1[0].bias.data / denominator

    model.layer2[0].weight.data = numerator * model.layer2[0].weight.data / denominator
    model.layer2[0].bias.data = numerator * model.layer2[0].bias.data / denominator

    model.layer3[0].weight.data = numerator * model.layer3[0].weight.data / denominator
    model.layer3[0].bias.data = numerator * model.layer3[0].bias.data / denominator

    model.fc1.weight.data = numerator * model.fc1.weight.data / denominator
    model.fc1.bias.data = numerator * model.fc1.bias.data / denominator

    model.fc2.weight.data = numerator * model.fc2.weight.data / denominator
    model.fc2.bias.data = numerator * model.fc2.bias.data / denominator


def add_model(*models):
    fed_add_model = CNN().to(device)

    fed_add_model.layer1[0].weight.data.fill_(0.0)
    fed_add_model.layer1[0].bias.data.fill_(0.0)

    fed_add_model.layer2[0].weight.data.fill_(0.0)
    fed_add_model.layer2[0].bias.data.fill_(0.0)

    fed_add_model.layer3[0].weight.data.fill_(0.0)
    fed_add_model.layer3[0].bias.data.fill_(0.0)

    fed_add_model.fc1.weight.data.fill_(0.0)
    fed_add_model.fc1.bias.data.fill_(0.0)

    fed_add_model.fc2.weight.data.fill_(0.0)
    fed_add_model.fc2.bias.data.fill_(0.0)

    for model in models:
        fed_add_model.layer1[0].weight.data += model.layer1[0].weight.data
        fed_add_model.layer1[0].bias.data += model.layer1[0].bias.data

        fed_add_model.layer2[0].weight.data += model.layer2[0].weight.data
        fed_add_model.layer2[0].bias.data += model.layer2[0].bias.data

        fed_add_model.layer3[0].weight.data += model.layer3[0].weight.data
        fed_add_model.layer3[0].bias.data += model.layer3[0].bias.data

        fed_add_model.fc1.weight.data += model.fc1.weight.data
        fed_add_model.fc1.bias.data += model.fc1.bias.data

        fed_add_model.fc2.weight.data += model.fc2.weight.data
        fed_add_model.fc2.bias.data += model.fc2.bias.data

    return fed_add_model


def sub_model(model1, model2):
    fed_sub_model = CNN().to(device)

    fed_sub_model.layer1[0].weight.data.fill_(0.0)
    fed_sub_model.layer1[0].bias.data.fill_(0.0)

    fed_sub_model.layer2[0].weight.data.fill_(0.0)
    fed_sub_model.layer2[0].bias.data.fill_(0.0)

    fed_sub_model.layer3[0].weight.data.fill_(0.0)
    fed_sub_model.layer3[0].bias.data.fill_(0.0)

    fed_sub_model.fc1.weight.data.fill_(0.0)
    fed_sub_model.fc1.bias.data.fill_(0.0)

    fed_sub_model.fc2.weight.data.fill_(0.0)
    fed_sub_model.fc2.bias.data.fill_(0.0)

    fed_sub_model.layer1[0].weight.data = model1.layer1[0].weight.data - model2.layer1[0].weight.data
    fed_sub_model.layer1[0].bias.data = model1.layer1[0].bias.data - model2.layer1[0].bias.data

    fed_sub_model.layer2[0].weight.data = model1.layer2[0].weight.data - model2.layer2[0].weight.data
    fed_sub_model.layer2[0].bias.data = model1.layer2[0].bias.data - model2.layer2[0].bias.data

    fed_sub_model.layer3[0].weight.data = model1.layer3[0].weight.data - model2.layer3[0].weight.data
    fed_sub_model.layer3[0].bias.data = model1.layer3[0].bias.data - model2.layer3[0].bias.data

    fed_sub_model.fc1.weight.data = model1.fc1.weight.data - model2.fc1.weight.data
    fed_sub_model.fc1.bias.data = model1.fc1.bias.data - model2.fc1.bias.data

    fed_sub_model.fc2.weight.data = model1.fc2.weight.data - model2.fc2.weight.data
    fed_sub_model.fc2.bias.data = model1.fc2.bias.data - model2.fc2.bias.data

    return fed_sub_model


def fed_avg(*models):
    fed_avg_model = CNN().to(device)

    fed_avg_model.layer1[0].weight.data.fill_(0.0)
    fed_avg_model.layer1[0].bias.data.fill_(0.0)

    fed_avg_model.layer2[0].weight.data.fill_(0.0)
    fed_avg_model.layer2[0].bias.data.fill_(0.0)

    fed_avg_model.layer3[0].weight.data.fill_(0.0)
    fed_avg_model.layer3[0].bias.data.fill_(0.0)

    fed_avg_model.fc1.weight.data.fill_(0.0)
    fed_avg_model.fc1.bias.data.fill_(0.0)

    fed_avg_model.fc2.weight.data.fill_(0.0)
    fed_avg_model.fc2.bias.data.fill_(0.0)

    for model in models:
        fed_avg_model.layer1[0].weight.data += model.layer1[0].weight.data
        fed_avg_model.layer1[0].bias.data += model.layer1[0].bias.data

        fed_avg_model.layer2[0].weight.data += model.layer2[0].weight.data
        fed_avg_model.layer2[0].bias.data += model.layer2[0].bias.data

        fed_avg_model.layer3[0].weight.data += model.layer3[0].weight.data
        fed_avg_model.layer3[0].bias.data += model.layer3[0].bias.data

        fed_avg_model.fc1.weight.data += model.fc1.weight.data
        fed_avg_model.fc1.bias.data += model.fc1.bias.data

        fed_avg_model.fc2.weight.data += model.fc2.weight.data
        fed_avg_model.fc2.bias.data += model.fc2.bias.data

    fed_avg_model.layer1[0].weight.data = fed_avg_model.layer1[0].weight.data / len(models)
    fed_avg_model.layer1[0].bias.data = fed_avg_model.layer1[0].bias.data / len(models)

    fed_avg_model.layer2[0].weight.data = fed_avg_model.layer2[0].weight.data / len(models)
    fed_avg_model.layer2[0].bias.data = fed_avg_model.layer2[0].bias.data / len(models)

    fed_avg_model.layer3[0].weight.data = fed_avg_model.layer3[0].weight.data / len(models)
    fed_avg_model.layer3[0].bias.data = fed_avg_model.layer3[0].bias.data / len(models)

    fed_avg_model.fc1.weight.data = fed_avg_model.fc1.weight.data / len(models)
    fed_avg_model.fc1.bias.data = fed_avg_model.fc1.bias.data / len(models)

    fed_avg_model.fc2.weight.data = fed_avg_model.fc2.weight.data / len(models)
    fed_avg_model.fc2.bias.data = fed_avg_model.fc2.bias.data / len(models)

    return fed_avg_model


def mitigate_fed_avg(model1, model2, alpha=0.5):
    output_model = CNN().to(device)

    output_model.layer1[0].weight.data.fill_(0.0)
    output_model.layer1[0].bias.data.fill_(0.0)

    output_model.layer2[0].weight.data.fill_(0.0)
    output_model.layer2[0].bias.data.fill_(0.0)

    output_model.layer3[0].weight.data.fill_(0.0)
    output_model.layer3[0].bias.data.fill_(0.0)

    output_model.fc1.weight.data.fill_(0.0)
    output_model.fc1.bias.data.fill_(0.0)

    output_model.fc2.weight.data.fill_(0.0)
    output_model.fc2.bias.data.fill_(0.0)

    output_model.layer1[0].weight.data = ((1 - alpha) * model1.layer1[0].weight.data) + (alpha * model2.layer1[0].weight.data)
    output_model.layer1[0].bias.data = ((1 - alpha) * model1.layer1[0].bias.data) + (alpha * model2.layer1[0].bias.data)

    output_model.layer2[0].weight.data = ((1 - alpha) * model1.layer2[0].weight.data) + (alpha * model2.layer2[0].weight.data)
    output_model.layer2[0].bias.data = ((1 - alpha) * model1.layer2[0].bias.data) + (alpha * model2.layer2[0].bias.data)

    output_model.layer3[0].weight.data = ((1 - alpha) * model1.layer3[0].weight.data) + (alpha * model2.layer3[0].weight.data)
    output_model.layer3[0].bias.data = ((1 - alpha) * model1.layer3[0].bias.data) + (alpha * model2.layer3[0].bias.data)

    output_model.fc1.weight.data = ((1 - alpha) * model1.fc1.weight.data) + (alpha * model2.fc1.weight.data)
    output_model.fc1.bias.data = ((1 - alpha) * model1.fc1.bias.data) + (alpha * model2.fc1.bias.data)

    output_model.fc2.weight.data = ((1 - alpha) * model1.fc2.weight.data) + (alpha * model2.fc2.weight.data)
    output_model.fc2.bias.data = ((1 - alpha) * model1.fc2.bias.data) + (alpha * model2.fc2.bias.data)


    return output_model


def test_label_predictions(model):
    model.eval()
    actuals = []
    predictions = []

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        test_data_loader = get_dataloader(type="all")

        for data, target in test_data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True)

            actuals.extend(target.view_as(prediction))
            predictions.extend(prediction)

            _, predicted = torch.max(output, 1)
            c = (predicted == target).squeeze()
            for i in range(4):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        for i in range(10):
            label_accuracy = round(100 * class_correct[i] / class_total[i], 2)
            print('accuracy of {0} : {1:.2f}'.format(labels[i], label_accuracy))

    return [i.item() for i in actuals], [i.item() for i in predictions]