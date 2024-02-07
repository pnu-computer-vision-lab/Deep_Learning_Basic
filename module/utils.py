import os
import random
import json
import numpy as np
from datetime import datetime

import torch

# run.py - parser
class Config:
    def __init__(self, args):
        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # if you have more than 1 GPU, may be define variable for other GPU (actually using pytorch lightning is better)
        self.epochs = args.epochs 
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.lr = args.lr
        self.seed = args.seed
        self.data_path = args.data_path
        self.save_dir = args.save_dir
        
        if args.exp_name is None:
            now = datetime.now()
            # self.exp_name = now.strftime("%Y-%m-%d %H:%M:%S")
            self.exp_name = now.strftime("%Y%m%d_%H%M%S")
        else:
            self.exp_name = args.exp_name
        
        self.log_dir = os.path.join(self.save_dir, self.exp_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self._save()
        
    def __str__(self):
        attr = vars(self)
        return "\n".join(f"{key}: {value}" for key, value in attr.items())
    
    def _save(self):
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            config = vars(self)
            json.dump(config, f, indent=4)
        

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# F1 score
# Accuracy

if __name__ == "__main__":
    pass