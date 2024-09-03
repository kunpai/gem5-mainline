#include "rnn/RNNInterconnect.hh"

#include <algorithm>
#include <random>

#include "debug/RNNInterconnect.hh"

namespace gem5
{

RNNInterconnect::RNNInterconnect(const Params &params)
    : ClockedObject(params),
      latency(params.latency),
      bandwidth(params.bandwidth)
{
    // Constructor implementation
}

void RNNInterconnect::initializeWeights(int fromSize, int toSize)
{
    // Initialize weights with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 0.1); // Mean 0, standard deviation 0.1

    weights.resize(fromSize);
    for (auto& row : weights) {
        row.resize(toSize);
        for (auto& weight : row) {
            weight = d(gen);
        }
    }
}

std::vector<float> RNNInterconnect::propagateSignals(
    const std::vector<float>& inputs)
{
    std::vector<float> outputs(weights[0].size(), 0.0f);

    // Simple matrix multiplication
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            outputs[j] += inputs[i] * weights[i][j];
        }
    }

    // Simulate latency by delaying the propagation
    // This is a simplified model and may need to be adjusted
    // based on your specific requirements
    schedule(new EventFunctionWrapper([this, outputs]() {
        // Here you would typically
        // notify the next layer or update some state
        // For now, we'll just print a message
        DPRINTF(RNNInterconnect,
            "Signals propagated after %f ticks\n", latency);
    }, name(), true), curTick() + latency * clockPeriod());

    // Apply bandwidth limitation
    // This is a simplified model and may need to be adjusted
    float totalSignal = std::accumulate(outputs.begin(), outputs.end(), 0.0f);
    if (totalSignal > bandwidth) {
        float scale = bandwidth / totalSignal;
        for (auto& output : outputs) {
            output *= scale;
        }
    }

    return outputs;
}

void RNNInterconnect::updateWeights(const std::vector<float>& inputs,
    const std::vector<float>& errors, float learningRate)
{
    // Implement backpropagation to update weights
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            weights[i][j] += learningRate * errors[j] * inputs[i];
        }
    }
}

} // namespace gem5
