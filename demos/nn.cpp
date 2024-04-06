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

    train_test train_temp = train_test_split(x, y, 0.2, 42);
    train_test val_test = train_test_split(train_temp.test_features, train_temp.test_targets, 0.5, 42);

    train_temp.train_features = min_max_scaler(train_temp.train_features);
    val_test.train_features = min_max_scaler(val_test.train_features);
    val_test.test_features = min_max_scaler(val_test.test_features);

    nn classifier = nn({4, 128, 3}, {RELU, SOFTMAX}, 0.01f);

    auto start = std::chrono::high_resolution_clock::now();

    classifier.train(train_temp.train_features, train_temp.train_targets, val_test.train_features,
                     val_test.train_targets);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    std::cout << "Time taken: " << duration.count() << " seconds\n";

    classifier.pred(val_test.test_features, val_test.test_targets);

    return 0;
}