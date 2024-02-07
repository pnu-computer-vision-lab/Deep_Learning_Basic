import torch
from torch.utils.tensorboard import SummaryWriter


def get_logger(name: str, **kwargs):
    if name == "tensorboard":
        # https://blog.naver.com/PostView.naver?blogId=wjddn9252&logNo=222371807209
        return SummaryWriter(**kwargs)
    else:
        ValueError(f"There is no {name} logging tool")
        

if __name__ == "__main__":
    pass

