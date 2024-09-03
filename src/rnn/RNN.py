from m5.objects.SimObject import SimObject
from m5.params import *


class RNN(SimObject):
    type = "RNN"
    cxx_header = "rnn/RNN.hh"
    cxx_class = "gem5::RNN"

    neurons = VectorParam.RNNNeuron([], "Neurons in the RNN")
    input_to_hidden_interconnect = Param.RNNInterconnect(
        "Interconnect between input and hidden layers"
    )
    hidden_to_hidden_interconnect = Param.RNNInterconnect(
        "Interconnect between hidden layers"
    )
    hidden_to_output_interconnect = Param.RNNInterconnect(
        "Interconnect between hidden and output layers"
    )
