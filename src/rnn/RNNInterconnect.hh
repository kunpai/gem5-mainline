#ifndef __RNN_INTERCONNECT_HH__
#define __RNN_INTERCONNECT_HH__

#include <vector>

#include "params/RNNInterconnect.hh"
#include "sim/clocked_object.hh"

namespace gem5
{

class RNNInterconnect : public ClockedObject
{
  private:
    std::vector<std::vector<float>> weights;
    float latency;
    float bandwidth;

  public:
    PARAMS(RNNInterconnect);
    RNNInterconnect(const Params &params);

    void initializeWeights(int fromSize, int toSize);
    std::vector<float> propagateSignals(const std::vector<float>& inputs);
    void updateWeights(const std::vector<float>& inputs,
    const std::vector<float>& errors, float learningRate);

    float getLatency() const { return latency; }
    float getBandwidth() const { return bandwidth; }
};

} // namespace gem5

#endif // __RNN_INTERCONNECT_HH__
