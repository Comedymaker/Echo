from train.replay_trainer import CombinerReplayTrainer
import random
import numpy as np
import torch
from utils.replay_buffer import ReplayBuffer

def set_random_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)


if __name__ == "__main__":
    set_random_seed(1057)
    replay_buffer = ReplayBuffer()
    replay_buffer.load(path="../autodl-tmp/replay_data/20250514101800/replay.pt", ratio=1)  # Load your replay buffer here
    replay_trainer = CombinerReplayTrainer()
    replay_trainer.refit(replay_buffer, num_epochs=15, batch_size=1)