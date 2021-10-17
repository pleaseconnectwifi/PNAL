import numpy as np
import time, os, math, operator, statistics, sys
# import tensorflow as tf
from random import Random

class Sample(object):
    def __init__(self, id, image, true_label):
        # image id
        self.id = id
        # image pixels
        self.image = image
        # image true label
        self.true_label = true_label
        # image corrupted label
        self.label = true_label

        ## for logging ###
        self.last_corrected_label = None
        self.corrected = False

    def toString(self):
        return "Id: " + str(self.id) + ", True Label: ", + str(self.true_label) + ", Corrupted Label: " + str(self.label)