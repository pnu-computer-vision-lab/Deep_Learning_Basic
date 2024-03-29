import argparse

from module.utils import Config
from module.trainer import Trainer


def run(config: Config):
    
    trainer = Trainer(config=config)
    trainer.setup()

    trainer.train()

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4) # if your os is Windows, then set 0
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./log")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model_name", type=str, default="CustomModel")

    args = parser.parse_args()
    
    config = Config(args)
    print(config)
    run(config)
