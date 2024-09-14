import m5
from m5.objects import *
from m5.params import *
from m5.util import addToPath

addToPath("../")

# Import our custom SimObjects
# from RNNNeuron import RNNNeuron
# from RNN import RNN
# from RNNTrainingSystem import RNNTrainingSystem

# Create the system
system = System()

# Set the clock frequency
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = "1GHz"
system.clk_domain.voltage_domain = VoltageDomain()

# Create the RNN

input_neurons = [
    RNNNeuron(
        neuron_id=i,
        layer_type="INPUT",
        num_inputs=1,
        learning_rate=0.01,
        seed=i,
    )
    for i in range(3)
]
hidden_neurons = [
    RNNNeuron(
        neuron_id=i + 3,
        layer_type="HIDDEN",
        num_inputs=3,
        num_recurrent_inputs=5,
        learning_rate=0.01,
        seed=i + 3,
    )
    for i in range(5)
]
output_neurons = [
    RNNNeuron(
        neuron_id=i + 8,
        layer_type="OUTPUT",
        num_inputs=5,
        learning_rate=0.01,
        seed=i + 8,
    )
    for i in range(2)
]

input_to_hidden = RNNInterconnect(latency=10.0, bandwidth=10.0)
hidden_to_hidden = RNNInterconnect(latency=10.0, bandwidth=10.0)
hidden_to_output = RNNInterconnect(latency=10.0, bandwidth=10.0)

# input_neuron = RNNNeuron(neuron_id=0, layer_type='INPUT')
# hidden_neuron = RNNNeuron(neuron_id=1, layer_type='HIDDEN', weights=[0.5])
# output_neuron = RNNNeuron(neuron_id=2, layer_type='OUTPUT', weights=[0.5])

rnn = RNN(
    neurons=input_neurons + hidden_neurons + output_neurons,
    input_to_hidden_interconnect=input_to_hidden,
    hidden_to_hidden_interconnect=hidden_to_hidden,
    hidden_to_output_interconnect=hidden_to_output,
)

# Create the RNN Training System
training_system = RNNTrainingSystem(
    rnn=rnn,
    training_data=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    target_outputs=[[0.4, 0.5], [0.7, 0.8]],
    max_epochs=30000,
)

# Add the training system to our system
system.training_system = training_system

# Create the root SimObject and start the simulation
root = Root(full_system=False, system=system)

# Instantiate the system
m5.instantiate()

print("Beginning simulation!")
exit_event = m5.simulate()
print("Exiting @ tick %i because %s" % (m5.curTick(), exit_event.getCause()))
