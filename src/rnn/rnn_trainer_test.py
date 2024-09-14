from m5 import (
    curTick,
    instantiate,
)
from m5.objects import (
    RNNNeuron,
    RNNTrainer,
    Root,
    SrcClockDomain,
    VoltageDomain,
)
from m5.simulate import simulate
from m5.stats import dump


# Create a test configuration
def run_test():
    neuron = RNNNeuron(
        numInputs=3,
        inputWeights=[0.5, -0.3, 0.8],
        recurrentWeights=[0.1, 0.1, 0.1],
        bias=0.2,
        activationFunction="tanh",
    )

    trainer = RNNTrainer(
        rnnNeuron=neuron,  # Pass the created neuron to the trainer
        numTrainingSteps=10,
        learningRate=0.2,
    )

    root = Root(full_system=False)
    root.trainer = trainer
    root.trainer.clk_domain = SrcClockDomain(
        clock="1GHz", voltage_domain=VoltageDomain()
    )

    # Instantiate the simulation
    instantiate()

    print("Starting the RNNTrainer simulation...")

    # Simulate for a sufficient number of ticks
    exit_event = simulate()
    print(
        f"Simulation ended at tick {curTick()} because {exit_event.getCause()}"
    )

    dump()


if __name__ == "__m5_main__":
    run_test()
