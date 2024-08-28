#ifndef __RNN_RNN_TRAINING_SYSTEM_HH__
#define __RNN_RNN_TRAINING_SYSTEM_HH__

#include <vector>

#include "params/RNNTrainingSystem.hh"
#include "rnn/RNN.hh"
#include "sim/clocked_object.hh"

namespace gem5
{

class RNNTrainingSystem : public ClockedObject
{
  private:
    RNN* rnn;
    std::vector<std::vector<float>> trainingData;
    std::vector<std::vector<float>> targetOutputs;
    int currentEpoch;
    int maxEpochs;

  public:
    PARAMS(RNNTrainingSystem);
    RNNTrainingSystem(const Params &params);

    void tick();
    void trainEpoch();
};

} // namespace gem5

#endif // __RNN_TRAINING_SYSTEM_HH__
