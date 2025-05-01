import os
from os import path as osp
import json

import numpy as np

def main():

    json.load('/workspace/dataset/objaverse_pcs_12500/000-000/000a3d9fa4ff4c888e71e698694eb0b0.npy', 'r') as f: