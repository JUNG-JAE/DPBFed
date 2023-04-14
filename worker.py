# ----------- System library ----------- #
import random
import warnings
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
from scipy.stats import beta
import pickle
# ---------- Learning library ---------- #
import torch

# ----------- Custom library ----------- #
from data_loader import worker_dataloader, source_dataloader
from system_utility import print_log, central_difference, create_directory
from learning_utility import get_network, model_weight_to_vector, gaussian_noise
from conf.global_settings import LEARNING_RATE, BATCH_SIZE, LOG_DIR, DATA_TYPE, POISONED_ATTACK_RATE, CDF_LOWER_BOUND

warnings.filterwarnings(action='ignore')


class Worker:
    def __init__(self, args, worker_id, global_round, logger, malicious=False):
        self.args = args
        self.worker_id = worker_id
        self.malicious_worker = malicious
        self.total_training_epoch = 0
        self.approve_list = {}
        # self.train_loader, self.test_loader = worker_dataloader(worker_id)
        self.train_loader, self.test_loader = source_dataloader()
        self.model = get_network(args)
        self.pre_global_model = get_network(args)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.total_batch = len(self.train_loader)
        self.logger = logger
        self.self_tracker = {}
        self.self_projection_variation = []
        self.peer_tracker = {}
        self.peer_projection_variation = {}
        self.peer_black_list = set()
        """ """
        if global_round - 1 > 0:
            self.model.load_state_dict(torch.load(f"./{LOG_DIR}/{DATA_TYPE}/{args.net}/global_model/G{global_round - 1}/aggregation.pt"), strict=True)
            self.pre_global_model.load_state_dict(torch.load(f"./{LOG_DIR}/{DATA_TYPE}/{args.net}/global_model/G{global_round - 1}/aggregation.pt"), strict=True)

            with open(f"{LOG_DIR}/{DATA_TYPE}/{args.net}/tracking_data/T{global_round - 1}/{self.worker_id}_tracker.pkl", 'rb') as f:
                self.peer_tracker = pickle.load(f)

            with open(f"{LOG_DIR}/{DATA_TYPE}/{args.net}/tracking_data/T{global_round - 1}/{self.worker_id}_self_tracker.pkl", 'rb') as f:
                self.self_tracker = pickle.load(f)

            with open(f"{LOG_DIR}/{DATA_TYPE}/{args.net}/tracking_data/T{global_round - 1}/{self.worker_id}_black_list.pkl", 'rb') as f:
                self.peer_black_list = pickle.load(f)

    def get_model(self):
        return self.model

    def train_model(self, epoch):
        self.total_training_epoch += epoch
        print_log(self.logger, f"Training model epoch: {epoch} | total training epoch: {self.total_training_epoch}")

        if self.args.gpu:
            self.model.to('cuda')

        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        for epoch in range(1, epoch + 1):
            progress = tqdm(total=len(self.train_loader.dataset), ncols=100)
            avg_loss = 0

            for images, labels in self.train_loader:
                if self.args.gpu:
                    images, labels = images.cuda(), labels.cuda()

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item() / len(self.train_loader.dataset)
                progress.update(BATCH_SIZE)

            print_log(self.logger, f"Epoch {epoch} average loss: {avg_loss:.4f}")
            progress.close()
        print_log(self.logger, " ")

        self.model.to('cpu')

        return self.model

    @torch.no_grad()
    def evaluate_model(self, model):

        if self.args.gpu:
            model.to('cuda')

        model.eval()

        test_loss = 0.0
        correct = 0.0
        total_samples = len(self.test_loader.dataset)

        for (images, labels) in self.test_loader:
            if self.args.gpu:
                images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            loss = self.loss_function(outputs, labels)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

        model.to('cpu')

        return correct.float() / total_samples

    def aggregation(self, *models):
        aggregation_model = get_network(self.args)
        aggregation_model_dict = OrderedDict()

        for index, model in enumerate(models):
            for layer in model.state_dict().keys():
                if index == 0:
                    aggregation_model_dict[layer] = 1 / len(models) * model.state_dict()[layer]
                else:
                    aggregation_model_dict[layer] += 1 / len(models) * model.state_dict()[layer]

        aggregation_model.load_state_dict(aggregation_model_dict)

        self.model = aggregation_model

        return aggregation_model

    def model_poisoning_attack(self, rate=POISONED_ATTACK_RATE):
        print_log(self.logger, "Model poisoning attack")

        poisoned_model = get_network(self.args)
        model_state_dict = self.model.state_dict()

        layers = [weights for weights in model_state_dict]
        number_of_poisoned_layer = int(len(layers) * rate)

        non_empty_layers = [layer for layer in layers if len(model_state_dict[layer].size()) != 0]
        number_of_poisoned_layer = min(number_of_poisoned_layer, len(non_empty_layers))

        sampled_layers = set(random.sample(non_empty_layers, number_of_poisoned_layer))

        poisoned_weights = {}

        for layer in layers:
            weights = model_state_dict[layer]

            if layer in sampled_layers:
                # poisoned_weight = weights + (gaussian_noise(weights.size()).cuda() if self.args.gpu else gaussian_noise(weights.size()))
                poisoned_weight = weights + gaussian_noise(weights.size())
            else:
                poisoned_weight = weights

            poisoned_weights[layer] = poisoned_weight

        poisoned_model.load_state_dict(poisoned_weights)
        self.model = poisoned_model

        return self.model

    def model_projection(self):
        # projcet local model -> previous global model local
        local_model_vector = model_weight_to_vector(self.model)
        global_model_vector = model_weight_to_vector(self.pre_global_model)

        local_model_dot_global_model = torch.dot(local_model_vector, global_model_vector)
        global_norm_sq = torch.dot(global_model_vector, global_model_vector)
        projection = local_model_dot_global_model / global_norm_sq * global_model_vector
        projection_length = torch.norm(projection)

        return projection_length

    def update_peer_change_rate(self):
        for worker_id, projection_values in self.peer_tracker.items():
            sorted_projection_dict = dict(sorted(projection_values.items(), key=lambda x: x[0]))
            sorted_projection_list = sorted(sorted_projection_dict.values())

            if len(sorted_projection_list) >= 2:
                worker_change_rate = np.round(central_difference(sorted_projection_list), 4)
                self.peer_projection_variation[worker_id] = worker_change_rate

        if len(self.self_tracker) >= 2:
            self_change_rate = np.round(central_difference(list(self.self_tracker.values())), 4)
            self.self_projection_variation = self_change_rate
            # print(f"self variation: {self.self_projection_variation}")

    def predict_malicious_worker(self):
        trackable_worker = [worker_id for worker_id, change_rate in self.peer_projection_variation.items() if len(change_rate) >= 2]

        if trackable_worker and len(self.self_projection_variation) >= 2:
            # worker_change_rate_mean = {worker_id: round(np.mean(change_rates), 4) for worker_id, change_rates in self.peer_projection_variation.items() if len(change_rates) >= 2}
            # total_mean = round(np.mean(list(worker_change_rate_mean.values())), 4)

            self_mean = np.mean(self.self_projection_variation)
            print(f"Self mean: {self_mean:.2f}")
            for worker_id, change_rates in self.peer_projection_variation.items():
                high_variation = sum(change_rate > self_mean for change_rate in change_rates)
                row_variation = sum(change_rate < self_mean for change_rate in change_rates)

                x = np.linspace(0, 1, 100)
                beta_cdf = beta.cdf(x, 1 + high_variation, 1 + row_variation)
                x_val = x[beta_cdf >= 0.5][0]

                if x_val > CDF_LOWER_BOUND and worker_id not in [self.peer_black_list, self.worker_id]:
                    self.peer_black_list.add(worker_id)
        else:
            print_log(self.logger, "< Not trackable >")

    def save_tracker(self, global_round):
        create_directory(LOG_DIR + "/" + DATA_TYPE + "/" + self.args.net + "/tracking_data/T" + str(global_round))
        with open(LOG_DIR + "/" + DATA_TYPE + "/" + self.args.net + "/tracking_data/T" + str(global_round) + "/" + self.worker_id + "_tracker.pkl", 'wb') as f:
            pickle.dump(self.peer_tracker, f)

        with open(LOG_DIR + "/" + DATA_TYPE + "/" + self.args.net + "/tracking_data/T" + str(global_round) + "/" + self.worker_id + "_self_tracker.pkl", 'wb') as f:
            pickle.dump(self.self_tracker, f)

        with open(LOG_DIR + "/" + DATA_TYPE + "/" + self.args.net + "/tracking_data/T" + str(global_round) + "/" + self.worker_id + '_black_list.pkl', 'wb') as f:
            pickle.dump(self.peer_black_list, f)
