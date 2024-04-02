// #include "datasets\englishspanish.h"
// #include "models\transformer.h"

// #include <iostream>
// #include <windows.h>

// int main()
// {
//     SetConsoleOutputCP(CP_UTF8);

//     EnglishSpanish englishSpanish = LoadEnglishSpanish();

//     for (int i = 0; i < englishSpanish.targetRaw.size(); ++i)
//         std::cout << englishSpanish.targetRaw[i] << " " << englishSpanish.contextRaw[i] << std::endl;

//     Transformer transformer = Transformer();

//     return 0;
// }

#include "datasets\iris.h"
#include "models\nn.h"
#include "preproc.h"

#include <chrono>

int main()
{
    Iris iris = LoadIris();
    Ten features = iris.features;
    Ten targets = iris.targets;

    targets = OneHot(targets, 3);

    TrainTest train_temp = TrainTestSplit(features, targets, 0.2, 42);
    TrainTest val_test = TrainTestSplit(train_temp.testFeatures, train_temp.testTargets, 0.5, 42);

    train_temp.trainFeatures = MinMaxScaler(train_temp.trainFeatures);
    val_test.trainFeatures = MinMaxScaler(val_test.trainFeatures);
    val_test.testFeatures = MinMaxScaler(val_test.testFeatures);

    NN nn = NN({4, 128, 3}, 0.01f);

    auto start = std::chrono::high_resolution_clock::now();

    nn.train(train_temp.trainFeatures, train_temp.trainTargets, val_test.trainFeatures, val_test.trainTargets);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    std::cout << "Time taken: " << duration.count() << " seconds\n";

    nn.predict(val_test.testFeatures, val_test.testTargets);

    return 0;
}