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

void NN::Train(const Tensor& xTrain, const Tensor& yTrain, const Tensor& xVal, const Tensor& yVal)
{
    weightBias = InitParameters();
    weightBiasMomentum = InitParameters();

    for (unsigned short i = 1; i <= epochs; ++i) {
        if (i > 10 && i < 20)      learningRate = 0.009f;
        else if (i > 20 && i < 30) learningRate = 0.005f;
        else                       learningRate = 0.001f;

        std::random_device rd;
        auto rdNum = rd();
        Tensor xShuffled = shuffle(xTrain, rdNum);
        Tensor yShuffled = shuffle(yTrain, rdNum);

        Tensor yBatch;
        TensorArray a;

        for (unsigned int j = 0; j < xTrain._shape.front(); j += batchSize) {
            Tensor xBatch = Slice(xShuffled, j, batchSize);
            yBatch = Slice(yShuffled, j, batchSize);

            a = ForwardPropagation(xBatch, weightBias.first, weightBias.second);
            
            std::vector<Tensor> dlDz, dlDw, dlDb;

            for (unsigned char k = layers.size() - 1; k > 0; --k) {
                if (k == layers.size() - 1)
                    dlDz.push_back(PrimeCategoricalCrossEntropy(yBatch, a.back()));
                else
                    dlDz.push_back(MatMul(dlDz[(layers.size() - 2) - k], Transpose(weightBias.first[k])) * PrimeRelu(a[k - 1]));
            }

            for (unsigned char k = layers.size() - 1; k > 0; --k) {
                if (k == 1)
                    dlDw.push_back(MatMul(Transpose(xBatch), dlDz[(layers.size() - 1) - k]));
                else
                    dlDw.push_back(MatMul(Transpose(a[k - 2]), dlDz[(layers.size() - 1) - k]));
            }

            for (unsigned char k = 0; k < layers.size() - 1; ++k)
                dlDb.push_back(Sum(dlDz[k], 0));

            for (unsigned char k = 0; k < layers.size() - 1; ++k) {
                dlDw[k] = ClipByValue(dlDw[k], -gradientClipThreshold, gradientClipThreshold);
                dlDb[k] = ClipByValue(dlDb[k], -gradientClipThreshold, gradientClipThreshold);
            }

            for (char k = layers.size() - 2; k >= 0; --k) {
                weightBiasMomentum.first[k] = momentum * weightBiasMomentum.first[k] - learningRate * dlDw[(layers.size() - 2) - k];
                weightBiasMomentum.second[k] = momentum * weightBiasMomentum.second[k] - learningRate * dlDb[(layers.size() - 2) - k];
            }

            for (char k = layers.size() - 2; k >= 0; --k) {
                weightBias.first[k] += weightBiasMomentum.first[k];
                weightBias.second[k] += weightBiasMomentum.second[k];
            }
        }
        
        std::cout << "Epoch " << i << "/" << epochs;
        std::cout << " - training loss: " << CategoricalCrossEntropy(yBatch, a.back()) << " - training accuracy: " << CategoricalAccuracy(yBatch, a.back());

        a = ForwardPropagation(xVal, weightBias.first, weightBias.second);
        std::cout << " - val loss: " << CategoricalCrossEntropy(yVal, a.back()) << " - val accuracy: " << CategoricalAccuracy(yVal, a.back());
        std::cout << std::endl;

        static unsigned char epochsWithoutImprovement = 0;
        static float bestValLoss = std::numeric_limits<float>::max();

        if (CategoricalCrossEntropy(yVal, a.back()) < bestValLoss) {
            bestValLoss = CategoricalCrossEntropy(yVal, a.back());
            epochsWithoutImprovement = 0;
        } else {
            epochsWithoutImprovement += 1;
        }

        if (epochsWithoutImprovement >= patience) {
            std::cout << std::endl << "Early stopping at epoch " << i + 1 << " as validation loss did not improve for " << static_cast<unsigned short>(patience) << " epochs." << std::endl;
            break;
        }
    }
}

void NN::Predict(const Tensor& xTest, const Tensor& yTest)
{
    auto a = ForwardPropagation(xTest, weightBias.first, weightBias.second);

    std::cout << std::endl;
    std::cout << "test loss: " << CategoricalCrossEntropy(yTest, a.back()) << " - test accuracy: " << CategoricalAccuracy(yTest, a.back());
    std::cout << std::endl << std::endl;

    std::cout << a.back() << std::endl << std::endl << yTest << std::endl;
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