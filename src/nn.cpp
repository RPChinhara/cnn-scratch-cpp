#include "nn.h"
#include "activations.h"
#include "arrays.h"
#include "derivatives.h"
#include "linalg.h"
#include "mathematics.h"
#include "random.h"

#include <random>

NN::NN(const std::vector<unsigned int>& layers, float learningRate)
{
    this->layers = layers;
    this->learningRate = learningRate;
}

void NN::Train(const Tensor& train_x, const Tensor& train_y, const Tensor& val_x, const Tensor& val_y)
{
    weightBias = InitParameters();
    weightBiasMomentum = InitParameters();

    for (unsigned short i = 1; i <= epochs; ++i) {
        if (i > 10 && i < 20)      learningRate = 0.009f;
        else if (i > 20 && i < 30) learningRate = 0.005f;
        else                       learningRate = 0.001f;

        std::random_device rd;
        auto rd_num = rd();
        Tensor x_shuffled = shuffle(train_x, rd_num);
        Tensor y_shuffled = shuffle(train_y, rd_num);

        Tensor y_batch;
        TensorArray  a;

        for (unsigned int j = 0; j < train_x._shape.front(); j += batchSize) {
            Tensor x_batch = Slice(x_shuffled, j, batch_size);
            y_batch = Slice(y_shuffled, j, batch_size);

            a = ForwardPropagation(x_batch, weightBias.first, weightBias.second);
            
            std::vector<Tensor> dl_dz, dl_dw, dl_db;

            for (unsigned char k = layers.size() - 1; k > 0; --k) {
                if (k == layers.size() - 1)
                    dl_dz.push_back(PrimeCategoricalCrossEntropy(y_batch, a.back()));
                else
                    dl_dz.push_back(MatMul(dl_dz[(layers.size() - 2) - k], Transpose(weightBias.first[k])) * PrimeRelu(a[k - 1]));
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
                dl_dw[k] = ClipByValue(dl_dw[k], -gradientClipThreshold, gradientClipThreshold);
                dl_db[k] = ClipByValue(dl_db[k], -gradientClipThreshold, gradientClipThreshold);
            }

            for (char k = layers.size() - 2; k >= 0; --k) {
                weightBiasMomentum.first[k] = momentum * weightBiasMomentum.first[k] - learningRate * dl_dw[(layers.size() - 2) - k];
                weightBiasMomentum.second[k] = momentum * weightBiasMomentum.second[k] - learningRate * dl_db[(layers.size() - 2) - k];
            }

            for (char k = layers.size() - 2; k >= 0; --k) {
                weightBias.first[k] += weightBiasMomentum.first[k];
                weightBias.second[k] += weightBiasMomentum.second[k];
            }
        }
        
        std::cout << "Epoch " << i << "/" << epochs;
        std::cout << " - training loss: " << CategoricalCrossEntropy(y_batch, a.back()) << " - training accuracy: " << CategoricalAccuracy(y_batch, a.back());

        a = ForwardPropagation(val_x, weightBias.first, weightBias.second);
        std::cout << " - val loss: " << CategoricalCrossEntropy(val_y, a.back()) << " - val accuracy: " << CategoricalAccuracy(val_y, a.back());
        std::cout << std::endl;

        static unsigned char epochs_without_improvement = 0;
        static float best_val_loss = std::numeric_limits<float>::max();

        if (CategoricalCrossEntropy(val_y, a.back()) < best_val_loss) {
            best_val_loss = CategoricalCrossEntropy(val_y, a.back());
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

void NN::Predict(const Tensor& test_x, const Tensor& test_y)
{
    auto a = ForwardPropagation(test_x, weightBias.first, weightBias.second);

    std::cout << std::endl;
    std::cout << "test loss: " << CategoricalCrossEntropy(test_y, a.back()) << " - test accuracy: " << CategoricalAccuracy(test_y, a.back());
    std::cout << std::endl << std::endl;

    std::cout << a.back() << std::endl << std::endl << test_y << std::endl;
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
        w.push_back(normal_distribution({ layers[i], layers[i + 1] }, 0.0f, 2.0f));
        b.push_back(Zeros({ 1, layers[i + 1] }));
    }

    return std::make_pair(w, b);
}