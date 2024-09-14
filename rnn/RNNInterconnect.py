from m5.objects.ClockedObject import ClockedObject
from m5.params import *
from m5.proxy import *


class RNNInterconnect(ClockedObject):
    type = "RNNInterconnect"
    cxx_header = "rnn/RNNInterconnect.hh"
    cxx_class = "gem5::RNNInterconnect"

    latency = Param.Float(1.0, "Latency of the interconnect")
    bandwidth = Param.Float(1.0, "Bandwidth of the interconnect")
