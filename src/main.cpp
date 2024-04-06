// #include "datas/enes.h"
// #include "mdls/trans.h"

// #include <iostream>
// #include <windows.h>

// int main()
// {
//     SetConsoleOutputCP(CP_UTF8);

//     EnEs en_es = load_en_es();

//     for (int i = 0; i < en_es.targetRaw.size(); ++i)
//         std::cout << en_es.targetRaw[i] << " " << en_es.contextRaw[i] << std::endl;

//     return 0;
// }

#include "datas/iris.h"
#include "mdls/nn.h"
#include "preproc.h"

#include <chrono>

int main()
{
    iris data = load_iris();
    ten x = data.x;
    ten y = data.y;

    y = one_hot(y, 3);

    train_test tr_te = train_test_split(x, y, 0.2, 42);
    train_test v_te = train_test_split(tr_te.test_features, tr_te.test_targets, 0.5, 42);

    tr_te.train_features = min_max_scaler(tr_te.train_features);
    v_te.train_features = min_max_scaler(v_te.train_features);
    v_te.test_features = min_max_scaler(v_te.test_features);

    nn classifier = nn({4, 128, 3}, {ACT_RELU, ACT_SOFTMAX}, 0.01f);

    auto start = std::chrono::high_resolution_clock::now();

    classifier.train(tr_te.train_features, tr_te.train_targets, v_te.train_features, v_te.train_targets);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    std::cout << "Time taken: " << duration.count() << " seconds\n";

    classifier.pred(v_te.test_features, v_te.test_targets);

    return 0;
}