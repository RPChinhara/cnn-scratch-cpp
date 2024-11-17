#include "acts.h"
#include "arrs.h"
#include "datas.h"
#include "linalg.h"
#include "losses.h"
#include "math.hpp"
#include "preproc.h"
#include "rd.h"
#include "tensor.h"

#include <cassert>
#include <chrono>
#include <functional>

using act_func = std::function<tensor(const tensor&)>;
using loss_func = std::function<float(const tensor&, const tensor&)>;
using metric_func = std::function<float(const tensor&, const tensor&)>;

class rnn {
  private:
    act_func activation;
    loss_func loss;
    float lr;
    size_t epochs = 150;
    size_t batch_size = 8317;

    size_t seq_length = 10;
    size_t input_size = 1;
    size_t hidden_size = 50;
    size_t output_size = 1;

    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-7f;
    size_t t = 0;

    tensor w_xh;
    tensor w_hh;
    tensor w_hy;
    tensor b_h;
    tensor b_y;

    tensor m_w_xh;
    tensor m_w_hh;
    tensor m_w_hy;
    tensor m_b_h;
    tensor m_b_y;

    tensor v_w_xh;
    tensor v_w_hh;
    tensor v_w_hy;
    tensor v_b_h;
    tensor v_b_y;

    enum Phase {
      TRAIN,
      TEST
    };

    std::tuple<std::vector<tensor>, std::vector<tensor>, std::vector<tensor>, std::vector<tensor>> forward(const tensor& x, enum Phase phase);

  public:
    rnn(const act_func &activation, const loss_func &loss, const float lr);
    void train(const tensor& x_train, const tensor& y_train);
    float evaluate(const tensor& x, const tensor& y);
    tensor predict(const tensor& x);
};

rnn::rnn(const act_func &activation, const loss_func &loss, const float lr) {
    this->activation = activation;
    this->loss = loss;
    this->lr = lr;

    w_xh = glorot_uniform(hidden_size, input_size);
    w_hh = glorot_uniform(hidden_size, hidden_size);
    w_hy = glorot_uniform(output_size, hidden_size);

    b_h = zeros({hidden_size, 1});
    b_y = zeros({output_size, 1});

    m_w_xh = zeros({hidden_size, input_size});
    m_w_hh = zeros({hidden_size, hidden_size});
    m_w_hy = zeros({output_size, hidden_size});

    m_b_h  = zeros({hidden_size, 1});
    m_b_y  = zeros({output_size, 1});

    v_w_xh = zeros({hidden_size, input_size});
    v_w_hh = zeros({hidden_size, hidden_size});
    v_w_hy = zeros({output_size, hidden_size});

    v_b_h  = zeros({hidden_size, 1});
    v_b_y  = zeros({output_size, 1});
}

void rnn::train(const tensor& x_train, const tensor& y_train) {
    for (auto i = 1; i <= epochs; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        auto [x_sequence, z_sequence, h_sequence, y_sequence] = forward(x_train, Phase::TRAIN);

        float error = loss(transpose(y_train), y_sequence.front());

        tensor d_loss_d_h_t = zeros({batch_size, hidden_size});

        tensor d_loss_d_w_xh = zeros({hidden_size, input_size});
        tensor d_loss_d_w_hh = zeros({hidden_size, hidden_size});
        tensor d_loss_d_b_h  = zeros({hidden_size, 1});

        float num_samples = static_cast<float>(y_train.shape.front());
        tensor d_loss_d_y = -2.0f / num_samples * (transpose(y_train) - y_sequence.front());

        for (auto j = seq_length; j > 0; --j) {
            if (j == seq_length) {
                tensor d_y_d_h_10 = w_hy;
                d_loss_d_h_t = matmul(transpose(d_loss_d_y), d_y_d_h_10);
            } else {
                d_loss_d_h_t = matmul(d_loss_d_h_t * transpose(relu_derivative(z_sequence[j])), w_hh);
            }

            d_loss_d_w_xh = d_loss_d_w_xh + matmul((transpose(d_loss_d_h_t) * relu_derivative(z_sequence[j - 1])), x_sequence[j - 1]);
            d_loss_d_w_hh = d_loss_d_w_hh + matmul((transpose(d_loss_d_h_t) * relu_derivative(z_sequence[j - 1])), transpose(h_sequence[j - 1]));

            d_loss_d_b_h  = d_loss_d_b_h + sum(transpose(d_loss_d_h_t) * relu_derivative(z_sequence[j - 1]), 1);
        }

        tensor d_loss_d_w_hy  = matmul(d_loss_d_y, transpose(h_sequence.back()));

        t += 1;

        m_w_xh = beta1 * m_w_xh + (1.0f - beta1) * d_loss_d_w_xh;
        m_w_hh = beta1 * m_w_hh + (1.0f - beta1) * d_loss_d_w_hh;
        m_w_hy = beta1 * m_w_hy + (1.0f - beta1) * d_loss_d_w_hy;

        m_b_h = beta1 * m_b_h + (1.0f - beta1) * d_loss_d_b_h;
        m_b_y = beta1 * m_b_y + (1.0f - beta1) * d_loss_d_y;

        v_w_xh = beta2 * v_w_xh + (1.0f - beta2) * square(d_loss_d_w_xh);
        v_w_hh = beta2 * v_w_hh + (1.0f - beta2) * square(d_loss_d_w_hh);
        v_w_hy = beta2 * v_w_hy + (1.0f - beta2) * square(d_loss_d_w_hy);

        v_b_h = beta2 * v_b_h + (1.0f - beta2) * square(d_loss_d_b_h);
        v_b_y = beta2 * v_b_y + (1.0f - beta2) * square(d_loss_d_y);

        tensor m_hat_w_xh = m_w_xh / (1.0f - powf(beta1, t));
        tensor m_hat_w_hh = m_w_hh / (1.0f - powf(beta1, t));
        tensor m_hat_w_hy = m_w_hy / (1.0f - powf(beta1, t));
        tensor m_hat_b_h = m_b_h / (1.0f - powf(beta1, t));
        tensor m_hat_b_y = m_b_y / (1.0f - powf(beta1, t));

        tensor v_hat_w_xh = v_w_xh / (1.0f - powf(beta2, t));
        tensor v_hat_w_hh = v_w_hh / (1.0f - powf(beta2, t));
        tensor v_hat_w_hy = v_w_hy / (1.0f - powf(beta2, t));
        tensor v_hat_b_h = v_b_h / (1.0f - powf(beta2, t));
        tensor v_hat_b_y = v_b_y / (1.0f - powf(beta2, t));

        w_xh = w_xh - lr * m_hat_w_xh / (sqrt(v_hat_w_xh) + epsilon);
        w_hh = w_hh - lr * m_hat_w_hh / (sqrt(v_hat_w_hh) + epsilon);
        w_hy = w_hy - lr * m_hat_w_hy / (sqrt(v_hat_w_hy) + epsilon);

        b_h = b_h - lr * m_hat_b_h / (sqrt(v_hat_b_h) + epsilon);
        b_y = b_y - lr * m_hat_b_y / (sqrt(v_hat_b_y) + epsilon);

        // w_xh = w_xh - lr * d_loss_d_w_xh;
        // w_hh = w_hh - lr * d_loss_d_w_hh;
        // w_hy = w_hy - lr * d_loss_d_w_hy;

        // b_h = b_h - lr * d_loss_d_b_h;
        // b_y = b_y - lr * d_loss_d_y;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remaining_ms = duration - seconds;

        std::cout << "Epoch " << i << "/" << epochs << std::endl << seconds.count() << "s " << remaining_ms.count() << "ms/step - loss: " << error << std::endl;
    }
}

float rnn::evaluate(const tensor& x, const tensor& y) {
    auto [x_sequence, z_sequence, h_sequence, y_sequence] = forward(x, Phase::TEST);
    return loss(transpose(y), y_sequence.front());
}

tensor rnn::predict(const tensor& x) {
    auto [x_sequence, z_sequence, h_sequence, y_sequence] = forward(x, Phase::TEST);
    return transpose(y_sequence.front());
}

std::tuple<std::vector<tensor>, std::vector<tensor>, std::vector<tensor>, std::vector<tensor>> rnn::forward(const tensor& x, enum Phase phase) {
    std::vector<tensor> x_sequence;
    std::vector<tensor> z_sequence;
    std::vector<tensor> h_sequence;
    std::vector<tensor> y_sequence;

    if (phase == Phase::TRAIN)
        batch_size = 8317;
    else
        batch_size = 2072;

    tensor h_t = zeros({hidden_size, batch_size});
    h_sequence.push_back(h_t);

    for (auto i = 0; i < seq_length; ++i) {
        size_t idx = i;

        tensor x_t = zeros({batch_size, input_size});

        for (auto j = 0; j < batch_size; ++j) {
            x_t[j] = x[idx];
            idx += seq_length;
        }

        tensor z_t = matmul(w_xh, transpose(x_t)) + matmul(w_hh, h_t) + b_h;
        h_t = activation(z_t);
        tensor y_t = matmul(w_hy, h_t) + b_y;

        x_sequence.push_back(x_t);
        z_sequence.push_back(z_t);
        h_sequence.push_back(h_t);

        if (i == seq_length - 1)
            y_sequence.push_back(y_t);
    }

    return std::make_tuple(x_sequence, z_sequence, h_sequence, y_sequence);
}

std::pair<tensor, tensor> create_sequences(const tensor& data, const size_t seq_length) {
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

int main() {
    tensor data = load_aapl();

    min_max_scaler scaler;
    scaler.fit(data);
    tensor scaled_data = scaler.transform(data);

    auto train_test = split(scaled_data, 0.2f);

    auto x_y_train = create_sequences(train_test.first, 10);
    auto x_y_test = create_sequences(train_test.second, 10);

    rnn model = rnn(relu, mean_squared_error, 0.01f);
    model.train(x_y_train.first, x_y_train.second);

    auto test_loss = model.evaluate(x_y_test.first, x_y_test.second);
    auto predict = scaler.inverse_transform(model.predict(x_y_test.first));

    x_y_test.second = scaler.inverse_transform(x_y_test.second);

    for (auto i = 0; i < x_y_test.second.size; ++i)
        std::cout << x_y_test.second[i] << " " << predict[i] << std::endl;

    std::cout << "Test  loss: " << test_loss << std::endl;

    return 0;
}