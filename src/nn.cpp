#include "nn.h"
#include "activation.h"
#include "array.h"
#include "derivative.h"
#include "linalg.h"
#include "mathematics.h"
#include "random.h"

#include <random>

NN::NN(const std::vector<unsigned int>& layers, float learning_rate)
{
    this->layers = layers;
    this->learning_rate = learning_rate;
}

void NN::Train(const Tensor& x_train, const Tensor& y_train, const Tensor& x_val, const Tensor& y_val)
{
    weight_bias = InitParameters();
    weight_bias_momentum = InitParameters();

    for (unsigned short i = 1; i <= epochs; ++i) {
        if (i > 10 && i < 20)      learning_rate = 0.009f;
        else if (i > 20 && i < 30) learning_rate = 0.005f;
        else                       learning_rate = 0.001f;

        std::random_device rd;
        auto rd_num = rd();
        Tensor x_shuffled = Shuffle(x_train, rd_num);
        Tensor y_shuffled = Shuffle(y_train, rd_num);

        Tensor y_batch;
        TensorArray a;

        for (unsigned int j = 0; j < x_train.shape.front(); j += batch_size) {
            Tensor x_batch = Slice(x_shuffled, j, batch_size);
            y_batch = Slice(y_shuffled, j, batch_size);

            a = ForwardPropagation(x_batch, weight_bias.first, weight_bias.second);
            
            std::vector<Tensor> dl_dz, dl_dw, dl_db;

            for (unsigned char k = layers.size() - 1; k > 0; --k) {
                if (k == layers.size() - 1)
                    dl_dz.push_back(PrimeCategoricalCrossEntropy(y_batch, a.back()));
                else
                    dl_dz.push_back(MatMul(dl_dz[(layers.size() - 2) - k], Transpose(weight_bias.first[k])) * PrimeRelu(a[k - 1]));
            }

            for (unsigned char k = layers.size() - 1; k > 0; --k) {
                if (k == 1)
                    dl_dw.push_back(MatMul(Transpose(x_batch), dl_dz[(layers.size() - 1) - k]));
                else
                    dl_dw.push_back(MatMul(Transpose(a[k - 2]), dl_dz[(layers.size() - 1) - k]));
            }

            for (unsigned char k = 0; k < layers.size() - 1; ++k)
                dl_db.push_back(Sum(dl_dz[k], 0));

            for (unsigned char k = 0; k < layers.size() - 1; ++k) {
                dl_dw[k] = ClipByValue(dl_dw[k], -gradient_clip_threshold, gradient_clip_threshold);
                dl_db[k] = ClipByValue(dl_db[k], -gradient_clip_threshold, gradient_clip_threshold);
            }

            for (char k = layers.size() - 2; k >= 0; --k) {
                weight_bias_momentum.first[k] = momentum * weight_bias_momentum.first[k] - learning_rate * dl_dw[(layers.size() - 2) - k];
                weight_bias_momentum.second[k] = momentum * weight_bias_momentum.second[k] - learning_rate * dl_db[(layers.size() - 2) - k];
            }

            for (char k = layers.size() - 2; k >= 0; --k) {
                weight_bias.first[k] += weight_bias_momentum.first[k];
                weight_bias.second[k] += weight_bias_momentum.second[k];
            }
        }
        
        std::cout << "Epoch " << i << "/" << epochs;
        std::cout << " - training loss: " << CategoricalCrossEntropy(y_batch, a.back()) << " - training accuracy: " << CategoricalAccuracy(y_batch, a.back());

        a = ForwardPropagation(x_val, weight_bias.first, weight_bias.second);
        std::cout << " - val loss: " << CategoricalCrossEntropy(y_val, a.back()) << " - val accuracy: " << CategoricalAccuracy(y_val, a.back());
        std::cout << std::endl;

        static unsigned char epochs_without_improvement = 0;
        static float best_val_loss = std::numeric_limits<float>::max();

        if (CategoricalCrossEntropy(y_val, a.back()) < best_val_loss) {
            best_val_loss = CategoricalCrossEntropy(y_val, a.back());
            epochs_without_improvement = 0;
        } else {
            epochs_without_improvement += 1;
        }

        if (epochs_without_improvement >= patience) {
            std::cout << std::endl << "Early stopping at epoch " << i + 1 << " as validation loss did not improve for " << static_cast<unsigned short>(patience) << " epochs." << std::endl;
            break;
        }
    }
}

void NN::Predict(const Tensor& x_test, const Tensor& y_test)
{
    auto a = ForwardPropagation(x_test, weight_bias.first, weight_bias.second);

    std::cout << std::endl;
    std::cout << "test loss: " << CategoricalCrossEntropy(y_test, a.back()) << " - test accuracy: " << CategoricalAccuracy(y_test, a.back());
    std::cout << std::endl << std::endl;

    std::cout << a.back() << std::endl << std::endl << y_test << std::endl;
}

TensorArray NN::ForwardPropagation(const Tensor& input, const TensorArray& w, const TensorArray& b)
{
    TensorArray z;
    TensorArray a;

    for (unsigned char i = 0; i < layers.size() - 1; ++i) {
        if (i == 0) {
            z.push_back((MatMul(input, w[i]) + b[i]));
            a.push_back((Relu(z[i])));
        } else {
            z.push_back((MatMul(a[i - 1], w[i]) + b[i]));
            if (i == 1)
                a.push_back((Softmax(z[i])));
        }
    }

    return a;
}

std::pair<TensorArray, TensorArray> NN::InitParameters()
{
    TensorArray w;
    TensorArray b;

    for (unsigned int i = 0; i < layers.size() - 1; ++i) {
        w.push_back(NormalDistribution({ layers[i], layers[i + 1] }, 0.0f, 6.0f));
        b.push_back(Zeros({ 1, layers[i + 1] }));
    }

    return std::make_pair(w, b);
}