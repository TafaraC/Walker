"""Offline subprocess for the new initializer and update recipes."""
import argparse
import math
import torch
from variable_training.common import read, save
from consolidated.models import initialize
from consolidated.worker import gail
from .update import train


def main():
    parser=argparse.ArgumentParser();parser.add_argument('job')
    args=parser.parse_args()
    from pathlib import Path
    path=Path(args.job);job=read(path);directory=path.parent
    if job['task']=='initialize':
        initialize(job,directory)
        state=torch.load(directory/'policy.pth',map_location='cpu',weights_only=True)
        state['actor_log_std'].fill_(math.log(job['exploration_std']))
        torch.save(state,directory/'policy.pth')
    elif job['recipe']=='legacy':
        gail(job,directory)
        metrics=read(directory/'metrics.json');metrics['recipe']='legacy'
        save(directory/'metrics.json',metrics)
    else:
        train(job,directory)


if __name__=='__main__':main()
