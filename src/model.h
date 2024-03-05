#pragma once

#include "action.h"
#include "tensor.h"

#include <vector>

class DQN
{
};

class DDQN
{
};

class FeedforwardNeuralNetwork
{
  public:
    FeedforwardNeuralNetwork(const std::vector<size_t> &layers, float const learningRate);
    void Train(const Tensor &xTrain, const Tensor &yTrain, const Tensor &xVal, const Tensor &yVal);
    void Predict(const Tensor &xTest, const Tensor &yTest);

  private:
    std::pair<std::vector<Tensor>, std::vector<Tensor>> InitParameters();
    std::vector<Tensor> ForwardPropagation(const Tensor &input, const std::vector<Tensor> &weights,
                                           const std::vector<Tensor> &biases);

    std::vector<size_t> layers;
    std::pair<std::vector<Tensor>, std::vector<Tensor>> weights_biases;
    std::pair<std::vector<Tensor>, std::vector<Tensor>> weights_biases_momentum;
    size_t batchSize = 10;
    size_t epochs = 200;
    float learningRate;
    float gradientClipThreshold = 8.0f;
    float momentum = 0.1f;
    size_t patience = 4;
};

class QLearning
{
  public:
    QLearning(size_t n_states, size_t n_actions, float learning_rate = 0.01f, float discount_factor = 0.5f,
              float exploration_rate = 1.0f, float exploration_decay = 0.995f, float exploration_min = 0.01f);
    Action ChooseAction(size_t state);
    void UpdateQtable(size_t state, Action action, float reward, size_t next_state, bool done);

    float exploration_rate;

  private:
    Tensor q_table;
    size_t n_states;
    size_t n_actions;
    float learning_rate;
    float discount_factor;
    float exploration_decay;
    float exploration_min;
};

class Transformer
{
};