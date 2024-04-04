#include "models\nn.h"
#include "datasets\iris.h"
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

    nn nn = nn({4, 128, 3}, 0.01f);

    auto start = std::chrono::high_resolution_clock::now();

    nn.Train(train_temp.trainFeatures, train_temp.trainTargets, val_test.trainFeatures, val_test.trainTargets);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    std::cout << "Time taken: " << duration.count() << " seconds\n";

    nn.Predict(val_test.testFeatures, val_test.testTargets);

    return 0;
}