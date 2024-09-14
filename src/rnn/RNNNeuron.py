from m5.objects.ClockedObject import ClockedObject
from m5.params import *
from m5.proxy import *


class RNNNeuron(ClockedObject):
    type = "RNNNeuron"
    cxx_header = "rnn/RNNNeuron.hh"
    cxx_class = "gem5::RNNNeuron"

    # Define parameters that map to the C++ class members
    numInputs = Param.Int("Number of input connections to the neuron")
    inputWeights = VectorParam.Float([], "Initial weights for the inputs")
    recurrentWeights = VectorParam.Float([], "Initial recurrent weights")
    bias = Param.Float(0.0, "Bias value of the neuron")
    activationFunction = Param.String(
        "tanh", "Activation function (e.g., 'tanh')"
    )
