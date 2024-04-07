// #include "act.h"
// #include "datas/enes.h"
// #include "mdls/trans.h"
// #include "ten.h"

// #include <iostream>
// #include <windows.h>

// int main()
// {
//     SetConsoleOutputCP(CP_UTF8);

//     EnEs en_es = load_en_es();

//     // for (int i = 0; i < en_es.targetRaw.size(); ++i)
//     //     std::cout << en_es.targetRaw[i] << " " << en_es.contextRaw[i] << std::endl;

//     // x = np.array([ 2.0, 1.0, 0.1 ])

//     ten x = ten({2.0, 1.0, 0.1}, {1, 3});
//     std::cout << act(x, ACT_SOFTMAX, DEV_CPU) << std::endl;

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
    train_test v_te = train_test_split(tr_te.x_test, tr_te.y_test, 0.5, 42);

    tr_te.x_train = min_max_scaler(tr_te.x_train);
    v_te.x_train = min_max_scaler(v_te.x_train);
    v_te.x_test = min_max_scaler(v_te.x_test);

    nn classifier = nn({4, 64, 32, 3}, {ACT_RELU, ACT_RELU, ACT_SOFTMAX}, 0.01f);

    auto start = std::chrono::high_resolution_clock::now();

    classifier.train(tr_te.x_train, tr_te.y_train, v_te.x_train, v_te.y_train);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    std::cout << "Time taken: " << duration.count() << " seconds\n";

    classifier.pred(v_te.x_test, v_te.y_test);

    auto aa = ten({1, 2, 3, 4, 5, 6, 4, 4, 4}, {3, 3});
    auto a2 = ten({1, 2, 3}, {3, 1});

    std::cout << aa - a2 << std::endl;

// Tensor(
// [[0.000000 1.000000 2.000000]
//  [2.000000 3.000000 4.000000]
//  [1.000000 1.000000 1.000000]], shape=(3, 3))

    return 0;
}