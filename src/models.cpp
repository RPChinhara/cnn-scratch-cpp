#include "models.h"
#include "activations.h"
#include "arrays.h"
#include "derivatives.h"
#include "linalg.h"
#include "losses.h"
#include "mathematics.h"
#include "metrics.h"
#include "random.h"

#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <string>

CNN2D::CNN2D(const std::vector<size_t> &filters, float const learning_rate)
{
    this->filters = filters;
    this->learning_rate = learning_rate;
}

void CNN2D::Train(const Tensor &xTrain, const Tensor &yTrain, const Tensor &xVal, const Tensor &yVal)
{
    // Tensor kernel = Zeros({3, 3});
    Tensor kernel = Tensor({1, -1, 1, 0, 1, 0, -1, 0, 1}, {3, 3});

    size_t kernelHeight = kernel.shape.front();
    size_t kernelWidth = kernel.shape.back();

    size_t inputHeight = xTrain.shape[1];
    size_t inputWidth = xTrain.shape[2];

    size_t outputHeight = inputHeight - kernelHeight + 1;
    size_t outputWidth = inputWidth - kernelWidth + 1;

    Tensor output = Zeros({outputHeight, outputWidth});

    // size_t idx = 0;

    // for (size_t i = 0; i < outputHeight; ++i)
    // {
    //     for (size_t j = 0; i < outputWidth; ++j)
    //     {
    //         // ouput[idx] =
    //     }
    // }

    std::cout << output << std::endl;
}

void CNN2D::Predict(const Tensor &xTest, const Tensor &yTest)
{
}

std::vector<Tensor> CNN2D::ForwardPropagation(const Tensor &input, const std::vector<Tensor> &kernel,
                                              const size_t stride)
{
    std::vector<Tensor> weights;

    return weights;
}

NN::NN(const std::vector<size_t> &layers, const float learningRate)
{
    this->layers = layers;
    this->learningRate = learningRate;
}

void NN::Train(const Tensor &x_train, const Tensor &y_train, const Tensor &x_val, const Tensor &y_val)
{
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    size_t rd_num;
    std::random_device rd;
    std::vector<std::string> buffer;
    Tensor x_shuffled;
    Tensor y_shuffled;
    Tensor x_batch;
    Tensor y_batch;
    std::vector<Tensor> activations;
    std::vector<Tensor> activations_val;
    std::vector<Tensor> dloss_dlogits, dloss_dweights, dloss_dbiases;

    weights_biases = InitParameters();
    weights_biases_momentum = InitParameters();

    for (size_t i = 1; i <= epochs; ++i)
    {
        startTime = std::chrono::high_resolution_clock::now();

        if (i > 10 && i < 20)
            learningRate = 0.009f;
        else if (i > 20 && i < 30)
            learningRate = 0.005f;
        else
            learningRate = 0.001f;

        rd_num = rd();
        x_shuffled = Shuffle(x_train, rd_num);
        y_shuffled = Shuffle(y_train, rd_num);

        for (size_t j = 0; j < x_train.shape.front(); j += batchSize)
        {
            assert(x_train.shape.front() >= batchSize && batchSize > 0);

            if (j + batchSize >= x_train.shape.front())
            {
                x_batch = Slice(x_shuffled, j, x_train.shape.front() - j);
                y_batch = Slice(y_shuffled, j, x_train.shape.front() - j);
            }
            else
            {
                x_batch = Slice(x_shuffled, j, batchSize);
                y_batch = Slice(y_shuffled, j, batchSize);
            }

            activations = ForwardPropagation(x_batch, weights_biases.first, weights_biases.second);

            size_t numLayers = layers.size() - 1;
            for (size_t k = numLayers; k > 0; --k)
            {
                if (k == numLayers)
                    dloss_dlogits.push_back(CategoricalCrossEntropyDerivative(y_batch, activations.back()));
                else
                    dloss_dlogits.push_back(MatMul(dloss_dlogits[(layers.size() - 2) - k],
                                                   Transpose(weights_biases.first[k]), Device::CPU) *
                                            ReluDerivative(activations[k - 1]));

                if (k == 1)
                    dloss_dweights.push_back(MatMul(Transpose(x_batch), dloss_dlogits[(numLayers)-k], Device::CPU));
                else
                    dloss_dweights.push_back(
                        MatMul(Transpose(activations[k - 2]), dloss_dlogits[(numLayers)-k], Device::CPU));

                dloss_dbiases.push_back(Sum(dloss_dlogits[(numLayers)-k], 0));

                dloss_dweights[(numLayers)-k] =
                    ClipByValue(dloss_dweights[(numLayers)-k], -gradientClipThreshold, gradientClipThreshold);
                dloss_dbiases[(numLayers)-k] =
                    ClipByValue(dloss_dbiases[(numLayers)-k], -gradientClipThreshold, gradientClipThreshold);

                weights_biases_momentum.first[k - 1] =
                    momentum * weights_biases_momentum.first[k - 1] - learningRate * dloss_dweights[(numLayers)-k];
                weights_biases_momentum.second[k - 1] =
                    momentum * weights_biases_momentum.second[k - 1] - learningRate * dloss_dbiases[(numLayers)-k];

                weights_biases.first[k - 1] += weights_biases_momentum.first[k - 1];
                weights_biases.second[k - 1] += weights_biases_momentum.second[k - 1];
            }

            dloss_dlogits.clear(), dloss_dweights.clear(), dloss_dbiases.clear();
        }

        activations_val = ForwardPropagation(x_val, weights_biases.first, weights_biases.second);

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remainingMilliseconds = duration - seconds;

        buffer.push_back("Epoch " + std::to_string(i) + "/" + std::to_string(epochs) + "\n" +
                         std::to_string(seconds.count()) + "s " + std::to_string(remainingMilliseconds.count()) +
                         "ms/step - loss: " + std::to_string(CategoricalCrossEntropy(y_batch, activations.back())) +
                         " - accuracy: " + std::to_string(CategoricalAccuracy(y_batch, activations.back())));
        buffer.back() += " - val_loss: " + std::to_string(CategoricalCrossEntropy(y_val, activations_val.back())) +
                         " - val_accuracy: " + std::to_string(CategoricalAccuracy(y_val, activations_val.back()));

        if (i % 10 == 0)
        {
            for (const auto &message : buffer)
                std::cout << message << '\n';
            buffer.clear();
        }

        // static size_t epochs_without_improvement = 0;
        // static float best_val_loss = std::numeric_limits<float>::max();
        // float loss = CategoricalCrossEntropy(y_val, activations.back());

        // if (loss < best_val_loss) {
        //     best_val_loss = loss;
        //     epochs_without_improvement = 0;
        // } else {
        //     epochs_without_improvement += 1;
        // }

        // if (epochs_without_improvement >= patience) {
        //     std::cout << '\n' << "Early stopping at epoch " << i + 1 << " as validation loss did not improve for " <<
        //     patience << " epochs." << '\n'; break;
        // }
    }
}

void NN::Predict(const Tensor &x_test, const Tensor &y_test)
{
    std::vector<Tensor> activations = ForwardPropagation(x_test, weights_biases.first, weights_biases.second);

    std::cout << '\n';
    std::cout << "test loss: " << std::to_string(CategoricalCrossEntropy(y_test, activations.back()))
              << " - test accuracy: " << std::to_string(CategoricalAccuracy(y_test, activations.back()));
    std::cout << "\n\n";

    std::cout << activations.back() << "\n\n" << y_test << '\n';
}

std::pair<std::vector<Tensor>, std::vector<Tensor>> NN::InitParameters()
{
    std::vector<Tensor> weights;
    std::vector<Tensor> biases;

    for (size_t i = 0; i < layers.size() - 1; ++i)
    {
        weights.push_back(NormalDistribution({layers[i], layers[i + 1]}, 0.0f, 0.2f));
        biases.push_back(Zeros({1, layers[i + 1]}));
    }

    return std::make_pair(weights, biases);
}

std::vector<Tensor> NN::ForwardPropagation(const Tensor &input, const std::vector<Tensor> &weights,
                                           const std::vector<Tensor> &biases)
{
    std::vector<Tensor> activations;

    for (size_t i = 0; i < layers.size() - 1; ++i)
    {
        if (i == 0)
        {
            activations.push_back(Relu(MatMul(input, weights[i], Device::CPU) + biases[i], Device::CPU));
        }
        else
        {
            if (i == layers.size() - 2)
                activations.push_back(Softmax(MatMul(activations[i - 1], weights[i], Device::CPU) + biases[i]));
            else
                activations.push_back(
                    Relu(MatMul(activations[i - 1], weights[i], Device::CPU) + biases[i], Device::CPU));
        }
    }

    return activations;
}

QLearning::QLearning(size_t n_states, size_t n_actions, float learning_rate, float discount_factor,
                     float exploration_rate, float exploration_decay, float exploration_min)
{
    this->n_states = n_states;
    this->n_actions = n_actions;
    this->learning_rate = learning_rate;
    this->discount_factor = discount_factor;
    this->exploration_rate = exploration_rate;
    this->exploration_decay = exploration_decay;
    this->exploration_min = exploration_min;
    q_table = Zeros({n_states, n_actions});
}

Action QLearning::ChooseAction(size_t state)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<> dis1(0.0f, 1.0f);

    if (dis1(rng) < exploration_rate)
    {
        std::uniform_int_distribution<> dis2(0, n_actions - 1);
        return static_cast<Action>(dis2(rng));
    }
    else
    {
        Tensor sliced_q_table = Slice(q_table, state, 1);
        size_t max_idx = 0;
        float max = std::numeric_limits<float>::lowest();

        for (size_t i = 0; i < sliced_q_table.size; ++i)
        {
            if (sliced_q_table[i] > max)
            {
                max = sliced_q_table[i];
                max_idx = i;
            }
        }

        return static_cast<Action>(max_idx);
    }
}

void QLearning::UpdateQtable(size_t state, Action action, float reward, size_t next_state, bool done)
{
    std::cout << Slice(q_table, state, 1) << "\n\n";

    Tensor sliced_q_table = Slice(q_table, next_state, 1);
    float next_max_q = std::numeric_limits<float>::lowest();

    for (size_t i = 0; i < sliced_q_table.size; ++i)
        if (sliced_q_table[i] > next_max_q)
            next_max_q = sliced_q_table[i];

    size_t idx = state == 0 ? action : (state * q_table.shape.back()) + action;
    q_table[idx] += learning_rate * (reward + discount_factor * next_max_q - q_table[idx]);

    if (exploration_rate <= exploration_min || done)
        exploration_rate = 1.0f;

    if (exploration_rate > exploration_min)
        exploration_rate *= exploration_decay;
}

Transformer::Transformer()
{
}

void Transformer::Train()
{
}

void Transformer::Predict()
{
}