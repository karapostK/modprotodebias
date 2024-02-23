import wandb
from train_probe import train_probe


def train_val_agent():
    # Initialization and gathering hyperparameters
    wandb.init(job_type='train/val')

    probe_config = {k: v for k, v in wandb.config.items() if k[0] != '_'}

    train_probe(probe_config, 'val')


train_val_agent()
