#include "acts.h"
#include "arrs.h"
#include "datas.h"
#include "linalg.h"
#include "losses.h"
#include "lyrs.h"
#include "math.h"
#include "rand.h"
#include "tensor.h"

#include <chrono>
#include <functional>
#include <random>

using act_func = std::function<tensor(const tensor&)>;
using loss_func = std::function<float(const tensor&, const tensor&)>;
using metric_func = std::function<float(const tensor&, const tensor&)>;

class nn {
  private:
    std::vector<size_t> lyrs;
    std::vector<act_func> activations;
    loss_func loss;
    metric_func metric;
    float lr;
    size_t epochs = 80;
    size_t batch_size = 10;
    float momentum = 0.1f;

    std::pair<std::vector<tensor>, std::vector<tensor>> w_b;
    std::pair<std::vector<tensor>, std::vector<tensor>> w_b_momentum;

    std::pair<std::vector<tensor>, std::vector<tensor>> init_params();
    std::pair<std::vector<tensor>, std::vector<tensor>> forward(const tensor& x, const std::vector<tensor> &w, const std::vector<tensor> &b);

  public:
    nn(const std::vector<size_t> &lyrs, const std::vector<act_func> &activations, const loss_func &loss, const metric_func &metric, const float lr);
    void train(const tensor& x_train, const tensor& y_train, const tensor& x_val, const tensor& y_val);
    float evaluate(const tensor& x, const tensor& y);
    tensor predict(const tensor& x);
};

nn::nn(const std::vector<size_t> &lyrs, const std::vector<act_func> &activations, const loss_func &loss, const metric_func &metric, const float lr) {
    this->lyrs = lyrs;
    this->activations = activations;
    this->loss = loss;
    this->metric = metric;
    this->lr = lr;

    w_b = init_params();
    w_b_momentum = init_params();
}

void nn::train(const tensor& x_train, const tensor& y_train, const tensor& x_val, const tensor& y_val) {
    for (auto i = 1; i <= epochs; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        if (10 <= i && i < 20)
            lr = 0.009f;
        else if (20 <= i && i < 30)
            lr = 0.005f;
        else if (30 <= i)
            lr = 0.001f;

        std::random_device rd;
        auto rd_state = rd();

        tensor x_shuffled = shuffle(x_train, rd_state);
        tensor y_shuffled = shuffle(y_train, rd_state);

        tensor x_batch;
        tensor y_batch;

        tensor y_pred;
        tensor y_pred_val;

        float accumulated_loss = 0.0f;

        for (auto j = 0; j < x_train.shape.front(); j += batch_size) {
            if (x_train.shape.front() <= j + batch_size) {
                x_batch = slice(x_shuffled, j, x_train.shape.front() - j);
                y_batch = slice(y_shuffled, j, x_train.shape.front() - j);
            } else {
                x_batch = slice(x_shuffled, j, batch_size);
                y_batch = slice(y_shuffled, j, batch_size);
            }

            auto [z, a] = forward(x_batch, w_b.first, w_b.second);
            y_pred = a.back();

            accumulated_loss += loss(y_batch, y_pred);

            std::vector<tensor> dl_dz, dl_dw, dl_db;

            for (auto k = lyrs.size() - 1; 0 < k; --k) {
                if (k == lyrs.size() - 1)
                    dl_dz.push_back(y_pred - y_batch);
                else
                    dl_dz.push_back(matmul(dl_dz[(lyrs.size() - 2) - k], transpose(w_b.first[k])) * relu_derivative(z[k - 1]));

                if (k == 1)
                    dl_dw.push_back(matmul(transpose(x_batch), dl_dz[(lyrs.size() - 1) - k]));
                else
                    dl_dw.push_back(matmul(transpose(a[k - 2]), dl_dz[(lyrs.size() - 1) - k]));

                dl_db.push_back(sum(dl_dz[(lyrs.size() - 1) - k], 0));

                dl_dw[(lyrs.size() - 1) - k] = clip_by_value(dl_dw[(lyrs.size() - 1) - k], -8.0f, 8.0f);
                dl_db[(lyrs.size() - 1) - k] = clip_by_value(dl_db[(lyrs.size() - 1) - k], -8.0f, 8.0f);

                w_b_momentum.first[k - 1] = momentum * w_b_momentum.first[k - 1] - lr * dl_dw[(lyrs.size() - 1) - k];
                w_b_momentum.second[k - 1] = momentum * w_b_momentum.second[k - 1] - lr * dl_db[(lyrs.size() - 1) - k];

                w_b.first[k - 1] += w_b_momentum.first[k - 1];
                w_b.second[k - 1] += w_b_momentum.second[k - 1];
            }

            dl_dz.clear(), dl_dw.clear(), dl_db.clear();
        }

        auto [z_val, a_val] = forward(x_val, w_b.first, w_b.second);
        y_pred_val = a_val.back();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remaining_ms = duration - seconds;

        std::cout << std::fixed << std::setprecision(5);
        std::cout << "Epoch " << i << "/" << epochs << std::endl;
        std::cout << seconds.count() << "s " << remaining_ms.count() << "ms/step - loss: " << accumulated_loss / ceil(x_train.shape.front() / batch_size) << " - accuracy: " << metric(y_batch, y_pred);
        std::cout << " - val_loss: " << loss(y_val, y_pred_val) << " - val_accuracy: " << metric(y_val, y_pred_val) << std::endl;
    }
}

float nn::evaluate(const tensor& x, const tensor& y) {
    auto [z, a] = forward(x, w_b.first, w_b.second);
    return loss(y, a.back());
}

tensor nn::predict(const tensor& x) {
    auto [z, a] = forward(x, w_b.first, w_b.second);
    return a.back();
}

std::pair<std::vector<tensor>, std::vector<tensor>> nn::init_params() {
    std::vector<tensor> w;
    std::vector<tensor> b;

    for (auto i = 0; i < lyrs.size() - 1; ++i) {
        w.push_back(normal_dist({lyrs[i], lyrs[i + 1]}, 0.0f, 0.2f));
        b.push_back(zeros({1, lyrs[i + 1]}));
    }

    return std::make_pair(w, b);
}

std::pair<std::vector<tensor>, std::vector<tensor>> nn::forward(const tensor& x, const std::vector<tensor> &w, const std::vector<tensor> &b) {
    std::vector<tensor> zs;
    std::vector<tensor> as;

    for (auto i = 0; i < lyrs.size() - 1; ++i) {
        if (i == 0) {
            tensor z = matmul(x, w[i]) + b[i];
            tensor a = activations[i](z);

            zs.push_back(z);
            as.push_back(a);
        } else {
            tensor z = matmul(as[i - 1], w[i]) + b[i];
            tensor a = activations[i](z);

            zs.push_back(z);
            as.push_back(a);
        }
    }

    return std::make_pair(zs, as);
}

float categorical_accuracy(const tensor& y_true, const tensor& y_pred) {
    tensor idx_true = argmax(y_true);
    tensor pred_idx = argmax(y_pred);
    float equal = 0.0f;

    for (auto i = 0; i < idx_true.size; ++i)
        if (idx_true[i] == pred_idx[i])
            ++equal;

    return equal / idx_true.size;
}

int main() {
    iris data = load_iris();
    tensor x = data.x;
    tensor y = data.y;

    y = one_hot(y, 3);

    auto x_train_test = split(x, 0.1f);
    auto y_train_test = split(y, 0.1f);

    min_max_scaler scaler;
    scaler.fit(x_train_test.first);
    x_train_test.first = scaler.transform(x_train_test.first);
    x_train_test.second = scaler.transform(x_train_test.second);

    nn model = nn({4, 64, 64, 3}, {relu, relu, softmax}, categorical_cross_entropy, categorical_accuracy, 0.01f);

    auto start = std::chrono::high_resolution_clock::now();

    model.train(x_train_test.first, y_train_test.first, x_train_test.second, y_train_test.second);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    std::cout << std::endl << "Time taken: " << duration.count() << " seconds" << std::endl << std::endl;

    auto test_loss = model.evaluate(x_train_test.second, y_train_test.second);
    auto pred = model.predict(x_train_test.second);

    std::cout << "Test loss: " << test_loss << std::endl;
    std::cout << std::endl << pred << std::endl;
    std::cout << std::endl << y_train_test.second << std::endl;

    return 0;
}