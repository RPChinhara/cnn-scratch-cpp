#pragma once

#include "losses.h"
#include "metrics.h"
#include "tensor.h"

#include <vector>

using TensorArray = std::vector<Tensor>;

class NN
{
public:
    NN(const std::vector<unsigned int>& layers, float learningRate);
    void Train(const Tensor& xTrain, const Tensor& ytrain, const Tensor& xVal, const Tensor& yVal);
    void Predict(const Tensor& xTest, const Tensor& yTest);

private:
    std::vector<unsigned int> layers;
    std::pair<TensorArray, TensorArray> weightBias;
    std::pair<TensorArray, TensorArray> weightBiasMomentum;
    
    unsigned short batchSize = 8;
    unsigned short epochs = 100;
    float learningRate;

    float gradientClipThreshold = 8.0f;
    float momentum = 0.1f;
    unsigned char patience = 12;
    
    TensorArray ForwardPropagation(const Tensor& input, const TensorArray& weight, const TensorArray& bias);
    std::pair<TensorArray, TensorArray> InitParameters();
};