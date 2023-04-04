import numpy as np
import time
import os
import sys

from .retinaface_post import get_retinaface_post, comute_retinaface_map
from .arcface_post import get_arcface_post, comute_arcface_acc
from .mobileface_post import parseList, get_mobileface_post, comute_mobface_acc


def get_faces(type, outputs, batch_size, params):
    return eval(type)(outputs, batch_size, params)




