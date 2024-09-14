#include "rnn/RNNTrainer.hh"

#include "debug/RNNTrainer.hh"
#include "sim/sim_events.hh"

namespace gem5
{

RNNTrainer::RNNTrainer(const Params &params) :
    ClockedObject(params),
    neuron(params.rnnNeuron),  // Use the externally passed neuron
    numTrainingSteps(params.numTrainingSteps),
    learningRate(params.learningRate),
    stats(*this)
{
    DPRINTF(RNNTrainer, "RNNTrainer created with %d training steps\n",
        numTrainingSteps);

    // Initialize some example training data and expected outputs
    trainingData = {
        {1.0, 0.5, -0.2},
        {0.3, 0.7, -0.1},
        {0.9, -0.4, 0.8}
    };
    expectedOutputs = {0.4, 0.5, 0.9};

    // Schedule the first training step
    scheduleTraining();
}

void RNNTrainer::scheduleTraining() {
    // Schedule the first training step at tick 100
    schedule(new EventFunctionWrapper([this] { trainStep(); },
        "Training Step"), 100);
}

void RNNTrainer::trainStep() {
    DPRINTF(RNNTrainer, "Performing a training step...\n");

    for (int i = 0; i < trainingData.size(); ++i) {
        float output = neuron->activate(trainingData[i]);
        float error = expectedOutputs[i] - output;
        neuron->learn(learningRate, error);

        DPRINTF(RNNTrainer,
        "Training step %d: input = %f, expected = %f, \
        output = %f, error = %f\n",
        i, trainingData[i][0], expectedOutputs[i], output, error);

        stats.numTrainingIterations++;
    }

    // Continue scheduling if there are more steps
    if (stats.numTrainingIterations.value() < numTrainingSteps) {
        schedule(new EventFunctionWrapper([this] { trainStep(); },
        "Training Step"),
        curTick() + 100);
    }
}

RNNTrainer::RNNTrainerStats::RNNTrainerStats(RNNTrainer &trainer) :
    statistics::Group(&trainer),
    trainer(trainer),
    ADD_STAT(numTrainingIterations,
    statistics::units::Count::get(),
    "Number of training iterations completed")
{
}

void RNNTrainer::RNNTrainerStats::regStats() {
    statistics::Group::regStats();
}

}  // namespace gem5
