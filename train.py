import tensorflow as tf
import numpy as np
from WaterDetection.Network.Net import Network

n = Network()
n.initialize()
n.topology5()
n.train()
