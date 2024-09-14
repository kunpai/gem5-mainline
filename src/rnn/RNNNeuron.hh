#ifndef __RNN_RNN_NEURON_HH__
#define __RNN_RNN_NEURON_HH__

#include <random>
#include <vector>

#include "debug/RNNNeuron.hh"
#include "params/RNNNeuron.hh"
#include "sim/clocked_object.hh"

namespace gem5
{


class RNNNeuron : public ClockedObject
{
  private:
    int numInputs;
    std::vector<float> inputWeights;
    std::vector<float> recurrentWeights;
    std::vector<float> inputs;
    std::vector<float> prevInputs;
    float bias;
    float output;
    std::string activationFunction;

    struct RNNNeuronStats : public statistics::Group
    {
      RNNNeuronStats(RNNNeuron& neuron);

      void regStats() override;

      RNNNeuron& neuron;

      statistics::Scalar numCycles;
      statistics::Scalar numActivations;
      statistics::Scalar numLearningUpdates;

    };

    RNNNeuronStats stats;

    public:
    PARAMS(RNNNeuron);
    RNNNeuron(const Params &params);

    float activate(const std::vector<float>& inputs);

    void learn(float learningRate, float error);
    float tanhActivation(float x) const;
    float tanhActivationDerivative(float x) const;

};

} // namespace gem5

#endif // __RNN_RNN_NEURON_HH__
