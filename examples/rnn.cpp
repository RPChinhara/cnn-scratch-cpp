#include "arrs.h"
#include "datas.h"
#include "lyrs.h"
#include "preproc.h"

std::pair<tensor, tensor> create_sequences(const tensor &data, const size_t seq_length) {
    tensor x = zeros({data.size - seq_length - 1, seq_length, 1});
    tensor y = zeros({data.size - seq_length - 1, 1});

    size_t idx = 0;
    for (auto i = 0; i < (data.size - seq_length - 1) * seq_length; ++i) {
        if (i % seq_length == 0 && i != 0)
            idx -= seq_length - 1;
        x[i] = data[idx];
        ++idx;
    }

    for (auto i = 0; i < data.size - seq_length - 1; ++i)
        y[i] = data[i + seq_length];

    return std::make_pair(x, y);
}

tensor hyperbolic_tangent(const tensor &z_t) {
    tensor h_t = z_t;

    for (auto i = 0; i < z_t.size; ++i)
        h_t.elem[i] = std::tanhf(z_t.elem[i]);

    return h_t;
}

float mean_squared_error(const tensor &y_true, const tensor &y_pred) {
    float sum = 0.0f;

    for (auto i = 0; i < y_true.size; ++i)
        sum += std::powf(y_true[i] - y_pred[i], 2.0f);

    return sum / y_true.size;
}

int main() {
    const float test_size = 0.2f;
    const size_t seq_length = 10;
    const size_t lr = 0.01f;

    tensor data = load_aapl();
    tensor scaled_data = min_max_scaler(data);
    auto train_test = split(scaled_data, test_size);

    auto x_y_train = create_sequences(train_test.first, seq_length);
    auto x_y_test = create_sequences(train_test.second, seq_length);

    rnn model = rnn(lr, hyperbolic_tangent, mean_squared_error);
    model.train(x_y_train.first, x_y_train.second, x_y_test.first, x_y_test.second);

    return 0;
}