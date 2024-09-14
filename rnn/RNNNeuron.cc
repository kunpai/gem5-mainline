#include <cmath>

#include "rnn/RNNNeuron.hh"

namespace gem5
{

RNNNeuron::RNNNeuron(const Params &params)
    : SimObject(params),
      id(params.neuron_id),
      layerType([&params]() {
          if (params.layer_type == "INPUT") return LayerType::INPUT;
          if (params.layer_type == "HIDDEN") return LayerType::HIDDEN;
          if (params.layer_type == "OUTPUT") return LayerType::OUTPUT;
          throw std::invalid_argument("Invalid layer type");
      }()),
      bias(params.bias),
      weights(params.weights.begin(), params.weights.end()),
      learningRate(params.learning_rate),
      rng(params.seed),
      dist(0.0, 0.1)
{
    weights.resize(params.num_inputs);
    std::generate(weights.begin(), weights.end(), [&]() { return dist(rng); });

    if (layerType == LayerType::HIDDEN) {
        recurrentWeights.resize(params.num_recurrent_inputs);
        std::generate(recurrentWeights.begin(),
        recurrentWeights.end(), [&]() { return dist(rng); });
    }

    bias = dist(rng);
}

void RNNNeuron::activate(const std::vector<float>& inputs,
const std::vector<float>& recurrentInputs)
{
    float sum = bias;
    for (size_t i = 0; i < inputs.size(); ++i) {
        sum += inputs[i] * weights[i];
    }
    if (layerType == LayerType::HIDDEN) {
        for (size_t i = 0; i < recurrentInputs.size()
            && i < recurrentWeights.size(); ++i) {
            sum += recurrentInputs[i] * recurrentWeights[i];
        }
    }
    state = tanh(sum);
}

void RNNNeuron::updateWeights(const std::vector<float>& inputs,
const std::vector<float>& recurrentInputs, float error)
{
    float delta = error * tanhDerivative(state);
    for (size_t i = 0; i < inputs.size() && i < weights.size(); ++i) {
        weights[i] += learningRate * delta * inputs[i];
    }
    if (layerType == LayerType::HIDDEN) {
        for (size_t i = 0; i < recurrentInputs.size()
            && i < recurrentWeights.size(); ++i) {
            recurrentWeights[i] += learningRate * delta * recurrentInputs[i];
        }
    }
    bias += learningRate * delta;
}

float RNNNeuron::tanh(float x) const
{
    return std::tanh(x);
}

float RNNNeuron::tanhDerivative(float x) const
{
    return 1.0 - x * x;
}

} // namespace gem5
