from m5.objects.ClockedObject import ClockedObject
from m5.params import *
from m5.proxy import *


class RNNTrainer(ClockedObject):
    type = "RNNTrainer"
    cxx_class = "gem5::RNNTrainer"
    cxx_header = "rnn/RNNTrainer.hh"

    # Define the parameters that map to the C++ class
    rnnNeuron = Param.RNNNeuron("Neuron to train")
    numTrainingSteps = Param.Int(100, "Number of training steps to perform")
    learningRate = Param.Float(0.01, "Learning rate for the neuron training")
