// #include "datasets\englishspanish.h"
// #include "models\transformer.h"

// #include <iostream>
// #include <windows.h>

// int main()
// {
//     SetConsoleOutputCP(CP_UTF8);

//     EnEs en_es = load_en_es();

//     for (int i = 0; i < en_es.targetRaw.size(); ++i)
//         std::cout << en_es.targetRaw[i] << " " << en_es.contextRaw[i] << std::endl;

//     Transformer transformer = Transformer();

//     return 0;
// }

#include "datasets\iris.h"
#include "models\nn.h"
#include "preproc.h"

#include <chrono>

int main()
{
    Iris iris = load_iris();
    Ten features = iris.features;
    Ten targets = iris.targets;

    targets = one_hot(targets, 3);

    TrainTest train_temp = train_test_split(features, targets, 0.2, 42);
    TrainTest val_test = train_test_split(train_temp.test_features, train_temp.test_targets, 0.5, 42);

    train_temp.train_features = min_max_scaler(train_temp.train_features);
    val_test.train_features = min_max_scaler(val_test.train_features);
    val_test.test_features = min_max_scaler(val_test.test_features);

    NN nn = NN({4, 128, 3}, 0.01f);

    auto start = std::chrono::high_resolution_clock::now();

    nn.train(train_temp.train_features, train_temp.train_targets, val_test.train_features, val_test.train_targets);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    std::cout << "Time taken: " << duration.count() << " seconds\n";

    nn.pred(val_test.test_features, val_test.test_targets);

    return 0;
}