import tensorflow as tf
import numpy as np
from WaterDetection.Network.Net import Network

n = Network()
''' Evaluating system with the chosen topology'''
# Batch size is 203 because it's a divisor of the total number of testing images. 45 iterations will take place using a batch size of 203
n.evaluate(203)
