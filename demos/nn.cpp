#include "mdls/nn.h"
#include "datas/iris.h"
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

    nn classifier = nn({4, 128, 3}, {ACT_RELU, ACT_SOFTMAX}, 0.01f);

    auto start = std::chrono::high_resolution_clock::now();

    classifier.train(tr_te.x_train, tr_te.y_train, v_te.x_train, v_te.y_train);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    std::cout << "Time taken: " << duration.count() << " seconds\n";

    classifier.pred(v_te.x_test, v_te.y_test);

    return 0;
}