""" Test controller """
import argparse
from os.path import join, exists
from utils.misc import RolloutGenerator
from models.controller import Controller
from utils.misc import LSIZE, RSIZE, ASIZE
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Where models are stored.')
args = parser.parse_args()

ctrl_file = join(args.logdir, 'ctrl', 'best.tar')

assert exists(ctrl_file),\
    "Controller was not trained..."

device = torch.device('cpu')

generator = RolloutGenerator(args.logdir, device, 1000)

controller = Controller(LSIZE, RSIZE, ASIZE) 
controller.load_state_dict(torch.load(ctrl_file, map_location=device))

with torch.no_grad():
    generator.rollout(None)