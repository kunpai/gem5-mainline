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

    u_int64_t totalPropagations;
    Tick totalLatency;
    float totalBandwidthUsed;
    Tick startTick;

    struct RNNInterconnectStats : public statistics::Group
    {
        RNNInterconnectStats(RNNInterconnect& interconn);

        void regStats() override;

        RNNInterconnect& interconn;

        statistics::Scalar latency;
        statistics::Scalar bandwidth;
        statistics::Formula seconds;
        statistics::Formula bandwidthGBps;
        statistics::Scalar ticks;
        statistics::Average actualLatency;
        statistics::Scalar propagationCount;

        void updateActualLatency(Tick actualLatencyTicks);
    };

    RNNInterconnectStats stats;

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
