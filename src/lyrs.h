#pragma once

#include "act.h"

#include <vector>

class ten;

class cnn2d
{
  public:
    cnn2d(const std::vector<size_t> &filters, float const lr);
    void train(const ten &xTrain, const ten &yTrain, const ten &xVal, const ten &yVal);
    void pred(const ten &xTest, const ten &yTest);

  private:
    std::vector<ten> forward_prop(const ten &input, const std::vector<ten> &kernel, const size_t stride);

    std::vector<size_t> filters;
    float lr;
};

class nn
{
  public:
    nn(const std::vector<size_t> &lyrs, const std::vector<act_enum> &act_types, float const lr);
    void train(const ten &x_train, const ten &y_train, const ten &x_val, const ten &y_val);
    void pred(const ten &x_test, const ten &y_test);

  private:
    std::pair<std::vector<ten>, std::vector<ten>> init_params();
    std::vector<ten> forward_prop(const ten &x, const std::vector<ten> &w, const std::vector<ten> &b);

    std::vector<size_t> lyrs;
    std::vector<act_enum> act_types;
    std::pair<std::vector<ten>, std::vector<ten>> w_b;
    std::pair<std::vector<ten>, std::vector<ten>> w_b_mom;
    std::vector<ten> a;
    size_t batch_size = 10;
    size_t epochs = 200;
    float lr;
    float grad_clip_threshold = 8.0f;
    float mom = 0.1f;
    size_t patience = 4;
};

ten embedding(const size_t in_dim, const size_t out_dim, const ten &ind);

/*
  #include "cnn2d.h"
  #include "mnist.h"
  #include "preproc.h"

  int main()
  {
      MNIST mnist = LoadMNIST();

      for (auto i = 0; i < 784; ++i)
      {

          if (i % 28 == 0)
              std::cout << std::endl;
          std::cout << mnist.trainImages[i] << "   ";
      }

      mnist.trainImages / 255.0f;
      mnist.testImages / 255.0f;

      mnist.trainLabels = OneHot(mnist.trainLabels, 10);
      mnist.testLabels = OneHot(mnist.testLabels, 10);

      cnn2d cnn2D = cnn2d({3, 128, 3}, 0.01f);
      cnn2D.Train(mnist.trainImages, mnist.trainLabels, mnist.testImages, mnist.testLabels);

      return 0;
  }

  #include "datas.h"
  #include "lyrs.h"
  #include "preproc.h"

  #include <chrono>

  int main()
  {
      iris data = load_iris();
      ten x = data.x;
      ten y = data.y;

      y = one_hot(y, 3);

      train_test train_temp = train_test_split(x, y, 0.2, 42);
      train_test val_test = train_test_split(train_temp.x_test, train_temp.y_test, 0.5, 42);

      train_temp.x_train = min_max_scaler(train_temp.x_train);
      val_test.x_train = min_max_scaler(val_test.x_train);
      val_test.x_test = min_max_scaler(val_test.x_test);

      nn classifier = nn({4, 64, 64, 3}, {RELU, RELU, SOFTMAX}, 0.01f);

      auto start = std::chrono::high_resolution_clock::now();

      classifier.train(train_temp.x_train, train_temp.y_train, val_test.x_train, val_test.y_train);

      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

      std::cout << "Time taken: " << duration.count() << " seconds\n";

      classifier.pred(val_test.x_test, val_test.y_test);

      return 0;
  }
*/