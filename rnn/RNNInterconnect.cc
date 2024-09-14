#include <algorithm>
#include <numeric>
#include <random>

#include "debug/RNNInterconnect.hh"
#include "rnn/RNNInterconnect.hh"
#include "sim/sim_events.hh"

namespace gem5
{

RNNInterconnect::RNNInterconnect(const Params &params)
    : ClockedObject(params),
      latency(params.latency),
      bandwidth(params.bandwidth),
      stats(*this),
      totalPropagations(0),
      totalLatency(0),
      totalBandwidthUsed(0),
      startTick(curTick())
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

std::vector<float> RNNInterconnect::propagateSignals
(const std::vector<float>& inputs)
{
    Tick startPropagationTick = curTick();

    std::vector<float> outputs(weights[0].size(), 0.0f);
    // Simple matrix multiplication
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            outputs[j] += inputs[i] * weights[i][j];
        }
    }

    // Calculate total signal (bandwidth usage)
    float totalSignal = std::accumulate(outputs.begin(),
    outputs.end(), 0.0f, [](float a, float b) { return a + std::abs(b); });

    // Apply bandwidth limitation
    if (totalSignal > bandwidth) {
        float scale = bandwidth / totalSignal;
        for (auto& output : outputs) {
            output *= scale;
        }
        totalSignal = bandwidth;
    }

    // Update stats
    totalPropagations++;
    totalLatency += latency;
    totalBandwidthUsed += totalSignal;

    // Simulate latency by delaying the propagation
    schedule(new EventFunctionWrapper([this, outputs,
    startPropagationTick]() {
        Tick actualLatency = curTick() - startPropagationTick;
        DPRINTF(RNNInterconnect,
        "Signals propagated after %llu ticks (expected: %llu)\n",
        actualLatency, latency);
        stats.updateActualLatency(actualLatency);
    }, name(), true), curTick() +
        latency * clockPeriod());

    return outputs;
}

void RNNInterconnect::updateWeights(const std::vector<float>& inputs,
const std::vector<float>& errors, float learningRate)
{
    // Implement backpropagation to update weights
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            weights[i][j] += learningRate
            * errors[j] * inputs[i];
        }
    }
}

RNNInterconnect::RNNInterconnectStats::RNNInterconnectStats(RNNInterconnect&
    _interconn)
    : statistics::Group(&_interconn),
      interconn(_interconn),
      ADD_STAT(latency,
      statistics::units::Tick::get(), "Average latency"),
      ADD_STAT(bandwidth,
      statistics::units::Ratio::get(), "Average bandwidth usage"),
      ADD_STAT(seconds,
      statistics::units::Second::get(), "Total simulation seconds"),
      ADD_STAT(bandwidthGBps,
      statistics::units::Ratio::get(), "Average bandwidth in GB/s"),
      ADD_STAT(ticks,
      statistics::units::Tick::get(), "Total simulation ticks"),
      ADD_STAT(actualLatency,
      statistics::units::Tick::get(), "Actual average latency"),
      ADD_STAT(propagationCount,
      statistics::units::Count::get(), "Total number of propagations")
{
    actualLatency.precision(2);
}

void RNNInterconnect::RNNInterconnectStats::regStats()
{
    using namespace statistics;

    latency = interconn.totalLatency / interconn.totalPropagations;
    bandwidth = interconn.totalBandwidthUsed /
        interconn.totalPropagations;
    seconds = (curTick() - interconn.startTick) / interconn.clockPeriod();
    bandwidthGBps = (interconn.totalBandwidthUsed / 1e9) / seconds;
    ticks = curTick() - interconn.startTick;
    propagationCount = interconn.totalPropagations;
}

void RNNInterconnect::RNNInterconnectStats::updateActualLatency(
    Tick actualLatencyTicks)
{
    actualLatency = actualLatencyTicks;
    propagationCount++;
}



} // namespace gem5
