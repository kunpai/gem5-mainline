from m5 import (
    curTick,
    instantiate,
)
from m5.objects import (
    RNNNeuron,
    Root,
    SrcClockDomain,
    VoltageDomain,
)
from m5.simulate import simulate
from m5.stats import (
    dump,
    reset,
)


def run_rnn_test():
    neuron = RNNNeuron(
        numInputs=3,
        inputWeights=[0.5, -0.3, 0.8],
        recurrentWeights=[0.1, 0.1, 0.1],
        bias=0.2,
        activationFunction="tanh",
    )

    root = Root(full_system=False)
    root.neuron = neuron
    root.neuron.clk_domain = SrcClockDomain(
        clock="1GHz", voltage_domain=VoltageDomain()
    )

    instantiate()

    print("RNNNeuron test running...")

    exit_event = simulate(100000)
    print(
        f"Simulation ended at tick {curTick()} because {exit_event.getCause()}"
    )

    dump()


if __name__ == "__m5_main__":
    run_rnn_test()
