#include "arrs.h"
#include "datas.h"
#include "lyrs.h"
#include "preproc.h"

std::pair<tensor, tensor> create_sequences(const tensor &data, const size_t seq_length) {
    tensor x = zeros({data.size - seq_length, seq_length, 1});
    tensor y = zeros({data.size - seq_length, 1});

    size_t idx = 0;
    for (auto i = 0; i < (data.size - seq_length) * seq_length; ++i) {
        if (i % seq_length == 0 && i != 0)
            idx -= seq_length - 1;
        x[i] = data[idx];
        ++idx;
    }

    for (auto i = 0; i < data.size - seq_length; ++i)
        y[i] = data[i + seq_length];

    return std::make_pair(x, y);
}

// tensor hyperbolic_tangent(const tensor &z_t) {
//     tensor h_t = z_t;

//     for (auto i = 0; i < z_t.size; ++i)
//         h_t.elem[i] = std::tanhf(z_t.elem[i]);

//     return h_t;
// }

tensor relu(const tensor &z) {
    tensor a = z;

    for (auto i = 0; i < z.size; ++i)
        a.elem[i] = std::fmax(0.0f, z.elem[i]);

    return a;
}

float mean_squared_error(const tensor &y_true, const tensor &y_pred) {
    float sum = 0.0f;
    float n = static_cast<float>(y_true.shape.back());

    for (auto i = 0; i < y_true.size; ++i)
        sum += std::powf(y_true[i] - y_pred[i], 2.0f);

    return sum / n;
}

int main() {
    tensor data = load_aapl();
    tensor scaled_data = min_max_scaler(data);
    auto train_test = split(scaled_data, 0.2f);

    auto x_y_train = create_sequences(train_test.first, 10);
    auto x_y_test = create_sequences(train_test.second, 10);

    // Everything much with google colab up to here! (delete this after everything is done)

    rnn model = rnn(relu, mean_squared_error, 0.01f);
    model.train(x_y_train.first, x_y_train.second, x_y_test.first, x_y_test.second);

    auto train_loss = model.evaluate(x_y_train.first, x_y_train.second);
    auto test_loss = model.evaluate(x_y_test.first, x_y_test.second);

    auto predict = model.predict(x_y_test.first);

    std::cout << "Train loss: " << train_loss << std::endl;
    std::cout << "Test  loss: " << test_loss << std::endl;
    // std::cout << std::endl << predict << std::endl;

    return 0;
}