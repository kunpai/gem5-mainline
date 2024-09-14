from m5.objects.ClockedObject import ClockedObject
from m5.params import *


class RNNTrainingSystem(ClockedObject):
    type = "RNNTrainingSystem"
    cxx_header = "rnn/RNNTrainingSystem.hh"
    cxx_class = "gem5::RNNTrainingSystem"

    rnn = Param.RNN("The RNN to train")

    # Define training_data as a vector of vectors of floats
    training_data = Param.VectorVectorFloat([], "Training data")

    # Define target_outputs as a vector of vectors of floats
    target_outputs = Param.VectorVectorFloat([], "Target outputs")

    max_epochs = Param.Int(1, "Maximum number of training epochs")
