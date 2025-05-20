from train.weight_network_trainer import WeightNetworkTrainer
import random
import numpy as np
import torch

def set_random_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)


if __name__ == "__main__":
    set_random_seed(1057)
    weightNetworkTrainer = WeightNetworkTrainer()
    weightNetworkTrainer.train_weight_network()
    weightNetworkTrainer.evaluate()