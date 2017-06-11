import tensorflow as tf
import numpy as np
from FloorDetectionNN.Network.Net import Network

n = Network()
n.initialize()
n.topology4()
n.train()
