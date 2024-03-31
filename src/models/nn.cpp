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
    std::vector<Tensor> a;
    std::vector<Tensor> a_val;
    std::vector<Tensor> dl_dz, dl_dw, dl_db;

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

            a = ForwardPropagation(x_batch, weights_biases.first, weights_biases.second);

            for (size_t k = numForwardBackProps; k > 0; --k)
            {
                if (k == numForwardBackProps)
                    dl_dz.push_back(dcce_da_da_dz(y_batch, a.back()));
                else
                    dl_dz.push_back(
                        MatMul(dl_dz[(layers.size() - 2) - k], Transpose(weights_biases.first[k]), Device::CPU) *
                        drelu_dz(a[k - 1]));

                if (k == 1)
                    dl_dw.push_back(MatMul(Transpose(x_batch), dl_dz[numForwardBackProps - k], Device::CPU));
                else
                    dl_dw.push_back(MatMul(Transpose(a[k - 2]), dl_dz[numForwardBackProps - k], Device::CPU));

                dl_db.push_back(Sum(dl_dz[numForwardBackProps - k], 0));

                dl_dw[numForwardBackProps - k] =
                    ClipByValue(dl_dw[numForwardBackProps - k], -gradientClipThreshold, gradientClipThreshold);
                dl_db[numForwardBackProps - k] =
                    ClipByValue(dl_db[numForwardBackProps - k], -gradientClipThreshold, gradientClipThreshold);

                weights_biases_momentum.first[k - 1] =
                    momentum * weights_biases_momentum.first[k - 1] - learningRate * dl_dw[numForwardBackProps - k];
                weights_biases_momentum.second[k - 1] =
                    momentum * weights_biases_momentum.second[k - 1] - learningRate * dl_db[numForwardBackProps - k];

                weights_biases.first[k - 1] += weights_biases_momentum.first[k - 1];
                weights_biases.second[k - 1] += weights_biases_momentum.second[k - 1];
            }

            dl_dz.clear(), dl_dw.clear(), dl_db.clear();
        }

        a_val = ForwardPropagation(x_val, weights_biases.first, weights_biases.second);

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remainingMilliseconds = duration - seconds;

        buffer.push_back("Epoch " + std::to_string(i) + "/" + std::to_string(epochs) + "\n" +
                         std::to_string(seconds.count()) + "s " + std::to_string(remainingMilliseconds.count()) +
                         "ms/step - loss: " + std::to_string(CategoricalCrossEntropy(y_batch, a.back())) +
                         " - accuracy: " + std::to_string(CategoricalAccuracy(y_batch, a.back())));
        buffer.back() += " - val_loss: " + std::to_string(CategoricalCrossEntropy(y_val, a_val.back())) +
                         " - val_accuracy: " + std::to_string(CategoricalAccuracy(y_val, a_val.back()));

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
    std::vector<Tensor> a = ForwardPropagation(x_test, weights_biases.first, weights_biases.second);

    std::cout << '\n';
    std::cout << "test loss: " << std::to_string(CategoricalCrossEntropy(y_test, a.back()))
              << " - test accuracy: " << std::to_string(CategoricalAccuracy(y_test, a.back()));
    std::cout << "\n\n";

    std::cout << a.back() << "\n\n" << y_test << '\n';
}

std::pair<std::vector<Tensor>, std::vector<Tensor>> NN::InitParameters()
{
    std::vector<Tensor> weight;
    std::vector<Tensor> bias;

    for (size_t i = 0; i < layers.size() - 1; ++i)
    {
        weight.push_back(NormalDistribution({layers[i], layers[i + 1]}, 0.0f, 0.2f));
        bias.push_back(Zeros({1, layers[i + 1]}));
    }

    return std::make_pair(weight, bias);
}

std::vector<Tensor> NN::ForwardPropagation(const Tensor &input, const std::vector<Tensor> &weight,
                                           const std::vector<Tensor> &bias)
{
    std::vector<Tensor> a;

    for (size_t i = 0; i < layers.size() - 1; ++i)
    {
        if (i == 0)
        {
            a.push_back(Relu(MatMul(input, weight[i], Device::CPU) + bias[i], Device::CPU));
        }
        else
        {
            if (i == layers.size() - 2)
                a.push_back(Softmax(MatMul(a[i - 1], weight[i], Device::CPU) + bias[i]));
            else
                a.push_back(Relu(MatMul(a[i - 1], weight[i], Device::CPU) + bias[i], Device::CPU));
        }
    }

    return a;
}