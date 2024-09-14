#ifndef __RNN_RNN_TRAINER_HH__
#define __RNN_RNN_TRAINER_HH__

#include "params/RNNTrainer.hh"
#include "rnn/RNNNeuron.hh"  // Include the RNNNeuron SimObject
#include "sim/clocked_object.hh"

namespace gem5
{

class RNNTrainer : public ClockedObject
{
  private:
    RNNNeuron *neuron;  // Pointer to the RNNNeuron object

    // Training parameters
    int numTrainingSteps;
    float learningRate;
    std::vector<std::vector<float>> trainingData;
    // Example input data for training
    std::vector<float> expectedOutputs;
    // Expected outputs for the training data

    struct RNNTrainerStats : public statistics::Group
    {
        RNNTrainerStats(RNNTrainer &trainer);

        void regStats() override;

        RNNTrainer &trainer;

        statistics::Scalar numTrainingIterations;
    };

    RNNTrainerStats stats;

  public:
    PARAMS(RNNTrainer);  // Required for gem5's parameter system
    RNNTrainer(const Params &params);

    // The function to initiate training
    void startTraining();

    // Simulate one training step (activating and learning)
    void trainStep();

    // Schedule periodic training steps
    void scheduleTraining();

    // Destructor
    ~RNNTrainer() override = default;
};

}  // namespace gem5

#endif  // __RNN_RNN_TRAINER_HH__
