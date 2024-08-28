#include "rnn/RNN.hh"

namespace gem5
{

RNN::RNN(const Params &params)
    : SimObject(params)
{
    for (auto neuron : params.neurons) {
        switch (neuron->getLayerType()) {
            case LayerType::INPUT:
                inputLayer.push_back(neuron);
                break;
            case LayerType::HIDDEN:
                hiddenLayer.push_back(neuron);
                break;
            case LayerType::OUTPUT:
                outputLayer.push_back(neuron);
                break;
        }
    }
    previousHiddenState.resize(hiddenLayer.size(), 0.0);
}

void RNN::processInput(const std::vector<float>& input)
{
    // Set input layer states
    for (size_t i = 0; i < input.size() && i < inputLayer.size(); ++i) {
        inputLayer[i]->setState(input[i]);
    }

    // Process hidden layer
    std::vector<float> hiddenInputs;
    for (auto inputNeuron : inputLayer) {
        hiddenInputs.push_back(inputNeuron->getState());
    }
    for (size_t i = 0; i < hiddenLayer.size(); ++i) {
        hiddenLayer[i]->activate(hiddenInputs, previousHiddenState);
        previousHiddenState[i] = hiddenLayer[i]->getState();
    }

    // Process output layer
    std::vector<float> outputInputs;
    for (auto hiddenNeuron : hiddenLayer) {
        outputInputs.push_back(hiddenNeuron->getState());
    }
    for (auto outputNeuron : outputLayer) {
        outputNeuron->activate(outputInputs, {});
    }
}

std::vector<float> RNN::getOutput() const
{
    std::vector<float> output;
    for (auto outputNeuron : outputLayer) {
        output.push_back(outputNeuron->getState());
    }
    return output;
}

void RNN::backpropagate(const std::vector<float>& target)
{
    // Calculate output layer errors
    std::vector<float> outputErrors;
    for (size_t i = 0; i < outputLayer.size(); ++i) {
        float error = target[i] - outputLayer[i]->getState();
        outputErrors.push_back(error);
    }

    // Update output layer weights
    std::vector<float> hiddenStates;
    for (auto hiddenNeuron : hiddenLayer) {
        hiddenStates.push_back(hiddenNeuron->getState());
    }
    for (size_t i = 0; i < outputLayer.size(); ++i) {
        outputLayer[i]->updateWeights(hiddenStates, {}, outputErrors[i]);
    }

    // Calculate hidden layer errors
    std::vector<float> hiddenErrors(hiddenLayer.size(), 0.0);
    for (size_t i = 0; i < hiddenLayer.size(); ++i) {
        for (size_t j = 0; j < outputLayer.size(); ++j) {
            // This is a simplification.
            // In a full implementation, we'd need access to the weights.
            hiddenErrors[i] += outputErrors[j] * 0.1;
            // 0.1 is a placeholder for the actual weight
        }
    }

    // Update hidden layer weights
    std::vector<float> inputStates;
    for (auto inputNeuron : inputLayer) {
        inputStates.push_back(inputNeuron->getState());
    }
    for (size_t i = 0; i < hiddenLayer.size(); ++i) {
        hiddenLayer[i]->updateWeights(inputStates,
        previousHiddenState, hiddenErrors[i]);
    }
}

} // namespace gem5
