#ifndef __RNN_RNN_NEURON_HH__
#define __RNN_RNN_NEURON_HH__

#include <random>
#include <vector>

#include "params/RNNNeuron.hh"
#include "sim/sim_object.hh"

namespace gem5
{

enum class LayerType
{
    INPUT,
    HIDDEN,
    OUTPUT
};

class RNNNeuron : public SimObject
{
  private:
    int id;
    LayerType layerType;
    float state;
    float bias;
    std::vector<float> weights;
    std::vector<float> recurrentWeights;
    float learningRate;

    std::mt19937 rng;
    std::normal_distribution<float> dist;

  public:
    PARAMS(RNNNeuron);
    RNNNeuron(const Params &params);

    int getId() const { return id; }
    LayerType getLayerType() const { return layerType; }
    void setState(float newState) { state = newState; }
    float getState() const { return state; }

    void activate(const std::vector<float>& inputs,
    const std::vector<float>& recurrentInputs);
    void updateWeights(const std::vector<float>& inputs,
    const std::vector<float>& recurrentInputs, float error);

    float tanh(float x) const;
    float tanhDerivative(float x) const;
};

} // namespace gem5

#endif // __RNN_RNN_NEURON_HH__
