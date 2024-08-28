from m5.objects.SimObject import SimObject
from m5.params import *


class RNN(SimObject):
    type = "RNN"
    cxx_header = "rnn/RNN.hh"
    cxx_class = "gem5::RNN"

    neurons = VectorParam.RNNNeuron([], "Neurons in the RNN")
