#include "nn.h"
#include "arrs.h"
#include "dev.h"
#include "diffs.h"
#include "linalg.h"
#include "losses.h"
#include "math.hpp"
#include "mets.h"
#include "rand.h"
#include "ten.h"

#include <cassert>
#include <chrono>
#include <random>

nn::nn(const std::vector<size_t> &lyrs, const std::vector<act_enum> &act_types, const float lr)
{
    this->lyrs = lyrs;
    this->act_types = act_types;
    this->lr = lr;
}

void nn::train(const ten &x_train, const ten &y_train, const ten &x_val, const ten &y_val)
{
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    size_t rd_num;
    std::random_device rd;
    std::vector<std::string> buff;
    ten x_shuffled;
    ten y_shuffled;
    ten x_batch;
    ten y_batch;
    std::vector<ten> a_val;
    std::vector<ten> dl_dz, dl_dw, dl_db;

    w_b = init_params();
    w_b_mom = init_params();

    for (size_t i = 1; i <= epochs; ++i)
    {
        start_time = std::chrono::high_resolution_clock::now();

        if (i > 10 && i < 20)
            lr = 0.009f;
        else if (i > 20 && i < 30)
            lr = 0.005f;
        else
            lr = 0.001f;

        rd_num = rd();
        x_shuffled = shuffle(x_train, rd_num);
        y_shuffled = shuffle(y_train, rd_num);

        for (size_t j = 0; j < x_train.shape.front(); j += batch_size)
        {
            assert(x_train.shape.front() >= batch_size && batch_size > 0);

            if (j + batch_size >= x_train.shape.front())
            {
                x_batch = slice(x_shuffled, j, x_train.shape.front() - j);
                y_batch = slice(y_shuffled, j, x_train.shape.front() - j);
            }
            else
            {
                x_batch = slice(x_shuffled, j, batch_size);
                y_batch = slice(y_shuffled, j, batch_size);
            }

            a = forward_prop(x_batch, w_b.first, w_b.second);

            for (size_t k = lyrs.size() - 1; k > 0; --k)
            {
                if (k == lyrs.size() - 1)
                    dl_dz.push_back(dl_da_da_dz(y_batch, a.back(), act_types.back()));
                else
                    dl_dz.push_back(matmul(dl_dz[(lyrs.size() - 2) - k], transpose(w_b.first[k]), DEV_CPU) *
                                    da_dz(a[k - 1], act_types[k - 1]));

                if (k == 1)
                    dl_dw.push_back(matmul(transpose(x_batch), dl_dz[(lyrs.size() - 1) - k], DEV_CPU));
                else
                    dl_dw.push_back(matmul(transpose(a[k - 2]), dl_dz[(lyrs.size() - 1) - k], DEV_CPU));

                dl_db.push_back(sum(dl_dz[(lyrs.size() - 1) - k], 0));

                dl_dw[(lyrs.size() - 1) - k] =
                    clip_by_value(dl_dw[(lyrs.size() - 1) - k], -grad_clip_threshold, grad_clip_threshold);
                dl_db[(lyrs.size() - 1) - k] =
                    clip_by_value(dl_db[(lyrs.size() - 1) - k], -grad_clip_threshold, grad_clip_threshold);

                w_b_mom.first[k - 1] = mom * w_b_mom.first[k - 1] - lr * dl_dw[(lyrs.size() - 1) - k];
                w_b_mom.second[k - 1] = mom * w_b_mom.second[k - 1] - lr * dl_db[(lyrs.size() - 1) - k];

                w_b.first[k - 1] += w_b_mom.first[k - 1];
                w_b.second[k - 1] += w_b_mom.second[k - 1];
            }

            dl_dz.clear(), dl_dw.clear(), dl_db.clear();
        }

        a_val = forward_prop(x_val, w_b.first, w_b.second);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remaining_ms = duration - seconds;

        buff.push_back("Epoch " + std::to_string(i) + "/" + std::to_string(epochs) + "\n" +
                       std::to_string(seconds.count()) + "s " + std::to_string(remaining_ms.count()) +
                       "ms/step - loss: " + std::to_string(categorical_cross_entropy(y_batch, a.back())) +
                       " - accuracy: " + std::to_string(categorical_acc(y_batch, a.back())));
        buff.back() += " - val_loss: " + std::to_string(categorical_cross_entropy(y_val, a_val.back())) +
                       " - val_accuracy: " + std::to_string(categorical_acc(y_val, a_val.back()));

        if (i % 10 == 0)
        {
            for (const auto &message : buff)
                std::cout << message << '\n';
            buff.clear();
        }

        // static size_t epochs_without_improvement = 0;
        // static float best_val_loss = std::numeric_limits<float>::max();
        // float loss = categorical_cross_entropy(y_val, a_val.back());

        // if (loss < best_val_loss)
        // {
        //     best_val_loss = loss;
        //     epochs_without_improvement = 0;
        // }
        // else
        // {
        //     epochs_without_improvement += 1;
        // }

        // if (epochs_without_improvement >= patience)
        // {
        //     std::cout << '\n'
        //               << "Early stopping at epoch " << i + 1 << " as validation loss did not improve for " <<
        //               patience
        //               << " epochs." << '\n';
        //     break;
        // }
    }
}

void nn::pred(const ten &x_test, const ten &y_test)
{
    a = forward_prop(x_test, w_b.first, w_b.second);

    std::cout << '\n';
    std::cout << "test loss: " << std::to_string(categorical_cross_entropy(y_test, a.back()))
              << " - test accuracy: " << std::to_string(categorical_acc(y_test, a.back()));
    std::cout << "\n\n";

    std::cout << a.back() << "\n\n" << y_test << '\n';
}

std::pair<std::vector<ten>, std::vector<ten>> nn::init_params()
{
    std::vector<ten> w;
    std::vector<ten> b;

    for (size_t i = 0; i < lyrs.size() - 1; ++i)
    {
        w.push_back(normal_dist({lyrs[i], lyrs[i + 1]}, 0.0f, 0.2f));
        b.push_back(zeros({1, lyrs[i + 1]}));
    }

    return std::make_pair(w, b);
}

std::vector<ten> nn::forward_prop(const ten &x, const std::vector<ten> &w, const std::vector<ten> &b)
{
    std::vector<ten> a;

    for (size_t i = 0; i < lyrs.size() - 1; ++i)
    {
        if (i == 0)
        {
            ten z = matmul(x, w[i], DEV_CPU) + b[i];
            a.push_back(act(z, act_types[i], DEV_CPU));
        }
        else
        {
            ten z = matmul(a[i - 1], w[i], DEV_CPU) + b[i];
            a.push_back(act(z, act_types[i], DEV_CPU));
        }
    }

    return a;
}