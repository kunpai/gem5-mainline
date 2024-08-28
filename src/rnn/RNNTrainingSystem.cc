#include "rnn/RNNTrainingSystem.hh"

#include "debug/RNNTrainingSystem.hh"
#include "sim/sim_exit.hh"

namespace gem5 {

RNNTrainingSystem::RNNTrainingSystem(const Params &params)
    : ClockedObject(params),
      rnn(params.rnn),
      trainingData(params.training_data),
      targetOutputs(params.target_outputs),
      currentEpoch(0),
      maxEpochs(params.max_epochs) {
    // Schedule the first tick
    schedule(new EventFunctionWrapper([this] { tick(); }, name(), true),
    curTick() + clockPeriod());
}

void RNNTrainingSystem::tick() {
    DPRINTF(RNNTrainingSystem, "RNN Training System tick at %llu\n",
    curTick());

    if (currentEpoch < maxEpochs) {
        trainEpoch();
        currentEpoch++;
        // Schedule the next tick
        schedule(new EventFunctionWrapper([this] { tick(); }, name(), true),
        curTick() + clockPeriod());
    } else {
        DPRINTF(RNNTrainingSystem,
            "RNN Training completed after %d epochs\n", maxEpochs);
        exitSimLoop("RNN Training completed");
    }
}

void RNNTrainingSystem::trainEpoch() {
    DPRINTF(RNNTrainingSystem, "Training epoch %d\n", currentEpoch);

    for (size_t i = 0; i < trainingData.size(); i++) {
        rnn->processInput(trainingData[i]);
        std::vector<float> output = rnn->getOutput();
        // Backpropagate and update weights
        rnn->backpropagate(targetOutputs[i]);
        // Print the output
        DPRINTF(RNNTrainingSystem, "Sample %zu - Output: ", i);
        for (float val : output) {
            DPRINTF(RNNTrainingSystem, "%f ", val);
        }
        DPRINTF(RNNTrainingSystem, "\n");
    }
}

} // namespace gem5
