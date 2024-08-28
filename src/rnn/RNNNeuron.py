from m5.objects.SimObject import SimObject
from m5.params import *


class RNNNeuron(SimObject):
    type = "RNNNeuron"
    cxx_header = "rnn/RNNNeuron.hh"
    cxx_class = "gem5::RNNNeuron"

    neuron_id = Param.Int(0, "Neuron ID")
    layer_type = Param.String("HIDDEN", "Layer type (INPUT, HIDDEN, OUTPUT)")
    bias = Param.Float(0.0, "Neuron bias")
    weights = VectorParam.Float([], "Synaptic weights")
    num_inputs = Param.Unsigned("Number of inputs to this neuron")
    num_recurrent_inputs = Param.Unsigned(
        0, "Number of recurrent inputs (for hidden neurons)"
    )
    learning_rate = Param.Float(0.01, "Learning rate for weight updates")
    seed = Param.Unsigned(42, "Seed for random number generation")
