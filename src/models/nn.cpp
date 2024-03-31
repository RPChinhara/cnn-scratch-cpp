#include "nn.h"
#include "activations.h"
#include "arrays.h"
#include "derivatives.h"
#include "device.h"
#include "linalg.h"
#include "losses.h"
#include "mathematics.h"
#include "metrics.h"
#include "random.h"
#include "tensor.h"

#include <cassert>
#include <chrono>
#include <random>

NN::NN(const std::vector<size_t> &layers, const float learningRate)
{
    this->layers = layers;
    this->numForwardBackProps = layers.size() - 1;
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
    std::pair<std::vector<Tensor>, Tensor> hiddensYpred;
    std::pair<std::vector<Tensor>, Tensor> hiddensYpredVal;
    std::vector<Tensor> dlossDy, dlossDhiddens, dlossDweights, dlossDbiases;

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

            hiddensYpred = ForwardPropagation(x_batch, weights_biases.first, weights_biases.second);

            // hidden1 = Relu(MatMul(inputs, weight1) + bias1)
            // yPred = Softmax(MatMul(hidden1, weight2) + bias2)
            // loss = Loss(yTrain, yPred)

            for (size_t k = numForwardBackProps; k > 0; --k)
            {
                if (k == numForwardBackProps)
                    dlossDhiddens.push_back(CategoricalCrossEntropyDerivative(y_batch, hiddensYpred.second));
                else
                    dlossDhiddens.push_back(MatMul(dlossDhiddens[(layers.size() - 2) - k],
                                                   Transpose(weights_biases.first[k]), Device::CPU) *
                                            ReluDerivative(hiddensYpred.first[k - 1]));

                if (k == 1)
                    dlossDweights.push_back(
                        MatMul(Transpose(x_batch), dlossDhiddens[numForwardBackProps - k], Device::CPU));
                else
                    dlossDweights.push_back(MatMul(Transpose(hiddensYpred.first[k - 2]),
                                                   dlossDhiddens[numForwardBackProps - k], Device::CPU));

                dlossDbiases.push_back(Sum(dlossDhiddens[numForwardBackProps - k], 0));

                dlossDweights[numForwardBackProps - k] = ClipByValue(
                    dlossDweights[numForwardBackProps - k], -gradientClipThreshold, gradientClipThreshold);
                dlossDbiases[numForwardBackProps - k] = ClipByValue(
                    dlossDbiases[numForwardBackProps - k], -gradientClipThreshold, gradientClipThreshold);

                weights_biases_momentum.first[k - 1] = momentum * weights_biases_momentum.first[k - 1] -
                                                       learningRate * dlossDweights[numForwardBackProps - k];
                weights_biases_momentum.second[k - 1] = momentum * weights_biases_momentum.second[k - 1] -
                                                        learningRate * dlossDbiases[numForwardBackProps - k];

                weights_biases.first[k - 1] += weights_biases_momentum.first[k - 1];
                weights_biases.second[k - 1] += weights_biases_momentum.second[k - 1];
            }

            dlossDhiddens.clear(), dlossDweights.clear(), dlossDbiases.clear();
        }

        hiddensYpredVal = ForwardPropagation(x_val, weights_biases.first, weights_biases.second);

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remainingMilliseconds = duration - seconds;

        buffer.push_back("Epoch " + std::to_string(i) + "/" + std::to_string(epochs) + "\n" +
                         std::to_string(seconds.count()) + "s " + std::to_string(remainingMilliseconds.count()) +
                         "ms/step - loss: " + std::to_string(CategoricalCrossEntropy(y_batch, hiddensYpred.second)) +
                         " - accuracy: " + std::to_string(CategoricalAccuracy(y_batch, hiddensYpred.second)));
        buffer.back() += " - val_loss: " + std::to_string(CategoricalCrossEntropy(y_val, hiddensYpredVal.second)) +
                         " - val_accuracy: " + std::to_string(CategoricalAccuracy(y_val, hiddensYpredVal.second));

        if (i % 10 == 0)
        {
            for (const auto &message : buffer)
                std::cout << message << '\n';
            buffer.clear();
        }

        // static size_t epochs_without_improvement = 0;
        // static float best_val_loss = std::numeric_limits<float>::max();
        // float loss = CategoricalCrossEntropy(y_val, hiddensOutputs.back());

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
    std::pair<std::vector<Tensor>, Tensor> hiddensYpred =
        ForwardPropagation(x_test, weights_biases.first, weights_biases.second);

    std::cout << '\n';
    std::cout << "test loss: " << std::to_string(CategoricalCrossEntropy(y_test, hiddensYpred.second))
              << " - test accuracy: " << std::to_string(CategoricalAccuracy(y_test, hiddensYpred.second));
    std::cout << "\n\n";

    std::cout << hiddensYpred.second << "\n\n" << y_test << '\n';
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

std::pair<std::vector<Tensor>, Tensor> NN::ForwardPropagation(const Tensor &input, const std::vector<Tensor> &weights,
                                                              const std::vector<Tensor> &biases)
{
    std::vector<Tensor> hiddens;
    Tensor yPred;

    for (size_t i = 0; i < layers.size() - 1; ++i)
    {
        if (i == 0)
        {
            hiddens.push_back(Relu(MatMul(input, weights[i], Device::CPU) + biases[i], Device::CPU));
        }
        else
        {
            if (i == layers.size() - 2)
                yPred = Softmax(MatMul(hiddens[i - 1], weights[i], Device::CPU) + biases[i]);
            else
                hiddens.push_back(Relu(MatMul(hiddens[i - 1], weights[i], Device::CPU) + biases[i], Device::CPU));
        }
    }

    return std::make_pair(hiddens, yPred);
}