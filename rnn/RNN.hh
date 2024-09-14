#ifndef __RNN_RNN_HH__
#define __RNN_RNN_HH__

#include <vector>

#include "params/RNN.hh"
#include "rnn/RNNInterconnect.hh"
#include "rnn/RNNNeuron.hh"
#include "sim/sim_object.hh"

namespace gem5
{

class RNN : public SimObject
{
  private:
    std::vector<RNNNeuron*> inputLayer;
    std::vector<RNNNeuron*> hiddenLayer;
    std::vector<RNNNeuron*> outputLayer;
    std::vector<float> previousHiddenState;

    RNNInterconnect* inputToHiddenInterconnect;
    RNNInterconnect* hiddenToHiddenInterconnect;
    RNNInterconnect* hiddenToOutputInterconnect;

  public:
    PARAMS(RNN);
    RNN(const Params &params);

    void processInput(const std::vector<float>& input);
    std::vector<float> getOutput() const;
    void backpropagate(const std::vector<float>& target);

};

} // namespace gem5

#endif // __RNN_RNN_HH__
