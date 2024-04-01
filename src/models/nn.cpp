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

NN::NN(const std::vector<size_t> &layers, const float lr)
{
    this->layers = layers;
    this->lr = lr;
}

void NN::train(const Tensor &x_train, const Tensor &y_train, const Tensor &x_val, const Tensor &y_val)
{
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    size_t rd_num;
    std::random_device rd;
    std::vector<std::string> buffer;
    Tensor x_shuffled;
    Tensor y_shuffled;
    Tensor x_batch;
    Tensor y_batch;
    std::vector<Tensor> a_val;
    std::vector<Tensor> dl_dz, dl_dw, dl_db;

    w_b = init_parameters();
    w_b_momentum = init_parameters();

    for (size_t i = 1; i <= epochs; ++i)
    {
        startTime = std::chrono::high_resolution_clock::now();

        if (i > 10 && i < 20)
            lr = 0.009f;
        else if (i > 20 && i < 30)
            lr = 0.005f;
        else
            lr = 0.001f;

        rd_num = rd();
        x_shuffled = Shuffle(x_train, rd_num);
        y_shuffled = Shuffle(y_train, rd_num);

        for (size_t j = 0; j < x_train.shape.front(); j += batch_size)
        {
            assert(x_train.shape.front() >= batch_size && batch_size > 0);

            if (j + batch_size >= x_train.shape.front())
            {
                x_batch = Slice(x_shuffled, j, x_train.shape.front() - j);
                y_batch = Slice(y_shuffled, j, x_train.shape.front() - j);
            }
            else
            {
                x_batch = Slice(x_shuffled, j, batch_size);
                y_batch = Slice(y_shuffled, j, batch_size);
            }

            a = forward_prop(x_batch, w_b.first, w_b.second);

            for (size_t k = layers.size() - 1; k > 0; --k)
            {
                if (k == layers.size() - 1)
                    dl_dz.push_back(dl_da_da_dz(y_batch, a.back(), SOFTMAX));
                else
                    dl_dz.push_back(MatMul(dl_dz[(layers.size() - 2) - k], Transpose(w_b.first[k]), Dev::CPU) *
                                    da_dz(a[k - 1], RELU));

                if (k == 1)
                    dl_dw.push_back(MatMul(Transpose(x_batch), dl_dz[(layers.size() - 1) - k], Dev::CPU));
                else
                    dl_dw.push_back(MatMul(Transpose(a[k - 2]), dl_dz[(layers.size() - 1) - k], Dev::CPU));

                dl_db.push_back(Sum(dl_dz[(layers.size() - 1) - k], 0));

                dl_dw[(layers.size() - 1) - k] =
                    ClipByValue(dl_dw[(layers.size() - 1) - k], -gradient_clip_threshold, gradient_clip_threshold);
                dl_db[(layers.size() - 1) - k] =
                    ClipByValue(dl_db[(layers.size() - 1) - k], -gradient_clip_threshold, gradient_clip_threshold);

                w_b_momentum.first[k - 1] = momentum * w_b_momentum.first[k - 1] - lr * dl_dw[(layers.size() - 1) - k];
                w_b_momentum.second[k - 1] =
                    momentum * w_b_momentum.second[k - 1] - lr * dl_db[(layers.size() - 1) - k];

                w_b.first[k - 1] += w_b_momentum.first[k - 1];
                w_b.second[k - 1] += w_b_momentum.second[k - 1];
            }

            dl_dz.clear(), dl_dw.clear(), dl_db.clear();
        }

        a_val = forward_prop(x_val, w_b.first, w_b.second);

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

void NN::predict(const Tensor &x_test, const Tensor &y_test)
{
    a = forward_prop(x_test, w_b.first, w_b.second);

    std::cout << '\n';
    std::cout << "test loss: " << std::to_string(CategoricalCrossEntropy(y_test, a.back()))
              << " - test accuracy: " << std::to_string(CategoricalAccuracy(y_test, a.back()));
    std::cout << "\n\n";

    std::cout << a.back() << "\n\n" << y_test << '\n';
}

std::pair<std::vector<Tensor>, std::vector<Tensor>> NN::init_parameters()
{
    std::vector<Tensor> w;
    std::vector<Tensor> b;

    for (size_t i = 0; i < layers.size() - 1; ++i)
    {
        w.push_back(NormalDistribution({layers[i], layers[i + 1]}, 0.0f, 0.2f));
        b.push_back(Zeros({1, layers[i + 1]}));
    }

    return std::make_pair(w, b);
}

std::vector<Tensor> NN::forward_prop(const Tensor &x, const std::vector<Tensor> &w, const std::vector<Tensor> &b)
{
    std::vector<Tensor> a;

    for (size_t i = 0; i < layers.size() - 1; ++i)
    {
        if (i == 0)
        {
            Tensor z = MatMul(x, w[i], Dev::CPU) + b[i];
            a.push_back(Relu(z, Dev::CPU));
        }
        else
        {
            if (i == layers.size() - 2)
            {
                Tensor z = MatMul(a[i - 1], w[i], Dev::CPU) + b[i];
                a.push_back(Softmax(z));
            }
            else
            {
                Tensor z = MatMul(a[i - 1], w[i], Dev::CPU) + b[i];
                a.push_back(Relu(z, Dev::CPU));
            }
        }
    }

    return a;
}