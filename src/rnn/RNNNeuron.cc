#include "rnn/RNNNeuron.hh"
#include <cmath>
#include "debug/RNNNeuron.hh"
#include "sim/sim_events.hh"

namespace gem5
{


RNNNeuron::RNNNeuron(const Params &params) : ClockedObject(params),
    numInputs(params.numInputs),
    inputWeights(params.inputWeights.begin(), params.inputWeights.end()),
    recurrentWeights(params.recurrentWeights.begin(),
        params.recurrentWeights.end()),
    bias(params.bias),
    output(0.0),
    activationFunction(params.activationFunction),
    stats(*this)
{
    // call regStats() to register statistics
    stats.regStats();
    DPRINTF(RNNNeuron, "Created RNNNeuron with %d inputs\n", numInputs);
    //  // For testing purposes, we perform activation here directly
    // std::vector<float> testInputs = {1.0, 0.5, -0.2};
    // float activation_output = activate(testInputs);
    // DPRINTF(RNNNeuron, "Activation output: %f\n", activation_output);

    // // For testing purposes, we also call learn here
    // float testError = 0.5;
    // float testLearningRate = 0.01;
    // learn(testLearningRate, testError);
    // DPRINTF(RNNNeuron, "Learning step 1 executed.\n");

    // // Round 2 of learning
    // testError = 0.2;
    // learn(testLearningRate, testError);
    // DPRINTF(RNNNeuron, "Learning step 2 executed.\n");
}

float RNNNeuron::activate(
    const std::vector<float>& inputs) {
    DPRINTF(RNNNeuron,
    "Activating RNNNeuron with %zu inputs\n",
    inputs.size());

    // Store current inputs
    this->inputs = inputs;

    // Compute the weighted sum for the input weights
    float weightedSum = 0.0;
    for (int i = 0; i < numInputs; ++i) {
        weightedSum += inputWeights[i] * inputs[i];
        stats.numCycles += 4; // 3 cycles for multiplication, 1 for addition
    }

    DPRINTF(RNNNeuron, "Weighted sum after input weights: %f\n", weightedSum);

    // Add contribution from the recurrent weights
    // (using prevInputs as the recurrent state)
    for (int i = 0; i < numInputs; ++i) {
        if (prevInputs.size() != numInputs) {
            // If this is the first timestep,
            // initialize the previous inputs to an array of zeros
            prevInputs = [this]()
                { std::vector<float> v(numInputs, 0.0);
                    return v; }();
        }
        weightedSum += recurrentWeights[i] * prevInputs[i];
        stats.numCycles += 4; // 3 cycles for multiplication, 1 for addition
    }

    DPRINTF(RNNNeuron, "Weighted sum after recurrent weights: %f\n",
        weightedSum);
    DPRINTF(RNNNeuron, "Bias: %f\n", bias);

    // Add the bias
    weightedSum += bias;
    DPRINTF(RNNNeuron, "Weighted sum after bias: %f\n", weightedSum);
    stats.numCycles += 1; // 1 cycle for addition

    DPRINTF(RNNNeuron, "Final weighted sum (including bias): %f\n",
        weightedSum);

    // Apply the activation function (e.g., tanh)
    if (activationFunction == "tanh") {
        output = tanhActivation(weightedSum);
    } else {
        // If more activation functions are needed, add here
        output = tanhActivation(weightedSum); // Default to tanh
    }
    stats.numCycles += 5; // Assume tanh takes about 5 cycles

    DPRINTF(RNNNeuron, "Output after activation: %f\n", output);

    // Update previous inputs (state) to current inputs for the next timestep
    prevInputs = inputs;

    stats.numActivations++;

    // Return the computed output
    return output;
}

void RNNNeuron::learn(float learningRate, float error) {
    DPRINTF(RNNNeuron, "Learning with rate %f and error %f\n",
        learningRate, error);

    // Compute the derivative of the activation function (for backpropagation)
    float derivative = 1.0; // Default derivative for non-linearities
    if (activationFunction == "tanh") {
        derivative = tanhActivationDerivative(output);
        // Use the output to compute the derivative
        stats.numCycles += 5;
        // Assume derivative calculation takes about 5 cycles
    }

    DPRINTF(RNNNeuron, "Activation function derivative: %f\n", derivative);

    // Update input weights using the error term
    // and the derivative of the activation
    for (int i = 0; i < numInputs; ++i) {
        float weightUpdate = learningRate * error * derivative * inputs[i];
        inputWeights[i] -= weightUpdate;
        stats.numCycles += 13;
        // 4 multiplications (12 cycles) and 1 subtraction
        DPRINTF(RNNNeuron,
        "Updated input weight %d: %f (change: %f)\n",
        i, inputWeights[i], weightUpdate);
    }

    // Update recurrent weights using the error term
    // and the previous inputs (state)
    for (int i = 0; i < numInputs; ++i) {
        float weightUpdate =
            learningRate * error * derivative * prevInputs[i];
        recurrentWeights[i] -= weightUpdate;
        stats.numCycles += 13;
        // 4 multiplications (12 cycles) and 1 subtraction
        DPRINTF(RNNNeuron, "Updated recurrent weight %d: %f (change: %f)\n",
        i, recurrentWeights[i], weightUpdate);
    }

    // Update bias
    float biasUpdate = learningRate * error * derivative;
    bias -= biasUpdate;
    stats.numCycles += 10; // 3 multiplications (9 cycles) and 1 subtraction
    DPRINTF(RNNNeuron, "Updated bias: %f (change: %f)\n", bias, biasUpdate);

    stats.numLearningUpdates++;
}

float RNNNeuron::tanhActivation(float x) const {
    return std::tanh(x);
}

float RNNNeuron::tanhActivationDerivative(float x) const {
    float tanh_x = tanhActivation(x);
    return 1.0f - tanh_x * tanh_x; // Derivative of tanh(x) is 1 - tanh(x)^2
}

RNNNeuron::RNNNeuronStats::RNNNeuronStats(RNNNeuron& neuron) :
    statistics::Group(&neuron), neuron(neuron),
    ADD_STAT(numCycles,
        statistics::units::Cycle::get(), "Number of cycles"),
    ADD_STAT(numActivations,
        statistics::units::Count::get(), "Number of activations"),
    ADD_STAT(numLearningUpdates,
        statistics::units::Count::get(), "Number of learning updates")
{
    numCycles = 0;
    numActivations = 0;
    numLearningUpdates = 0;
}


void RNNNeuron::RNNNeuronStats::regStats() {
    using namespace statistics;
    statistics::Group::regStats();

}

} // namespace gem5
