#include "debug/RNNTrainingSystem.hh"
#include "rnn/RNNTrainingSystem.hh"
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
    DPRINTF(RNNTrainingSystem,
        "RNN Training System tick at %llu, epoch %d/%d\n",
        curTick(), currentEpoch + 1, maxEpochs);

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
    DPRINTF(RNNTrainingSystem, "Training epoch %d, total samples: %zu\n",
            currentEpoch + 1, trainingData.size());

    for (size_t i = 0; i < trainingData.size(); i++) {
        DPRINTF(RNNTrainingSystem, "Processing sample %zu/%zu in epoch %d\n",
                i + 1, trainingData.size(), currentEpoch + 1);

        rnn->processInput(trainingData[i]);
        std::vector<float> output = rnn->getOutput();
        // Backpropagate and update weights
        rnn->backpropagate(targetOutputs[i]);

        // Print the output
        DPRINTF(RNNTrainingSystem, "Sample %zu - Output: [", i);
        for (float val : output) {
            DPRINTF(RNNTrainingSystem, "%f ", val);
        }
        DPRINTF(RNNTrainingSystem, "]\n");

        // Optionally print target outputs and error
        DPRINTF(RNNTrainingSystem, "Sample %zu - Target: [", i);
        for (float target : targetOutputs[i]) {
            DPRINTF(RNNTrainingSystem, "%f ", target);
        }
        DPRINTF(RNNTrainingSystem, "]\n");
    }
}

} // namespace gem5
