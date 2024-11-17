#include "acts.h"
#include "arrs.h"
#include "datas.h"
#include "linalg.h"
#include "losses.h"
#include "math.hpp"
#include "preproc.h"
#include "rd.h"
#include "tensor.h"

#include <array>
#include <cassert>
#include <chrono>
#include <functional>

using act_func = std::function<tensor(const tensor&)>;
using loss_func = std::function<float(const tensor&, const tensor&)>;
using metric_func = std::function<float(const tensor&, const tensor&)>;

class lstm {
  private:
    loss_func loss;
    float lr;
    size_t batch_size;
    size_t epochs = 250;

    size_t seq_length = 10;
    size_t input_size = 1;
    size_t hidden_size = 50;
    size_t output_size = 1;

    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-7f;
    size_t t = 0;

    tensor w_f;
    tensor w_i;
    tensor w_c;
    tensor w_o;
    tensor w_y;

    tensor b_f;
    tensor b_i;
    tensor b_c;
    tensor b_o;
    tensor b_y;

    tensor m_w_f;
    tensor m_w_i;
    tensor m_w_c;
    tensor m_w_o;
    tensor m_w_y;

    tensor m_b_f;
    tensor m_b_i;
    tensor m_b_c;
    tensor m_b_o;
    tensor m_b_y;

    tensor v_w_f;
    tensor v_w_i;
    tensor v_w_c;
    tensor v_w_o;
    tensor v_w_y;

    tensor v_b_f;
    tensor v_b_i;
    tensor v_b_c;
    tensor v_b_o;
    tensor v_b_y;

    enum Phase {
      TRAIN,
      TEST
    };

    std::array<std::vector<tensor>, 12> forward(const tensor& x, enum Phase phase);

  public:
    lstm(const loss_func &loss, const float lr);
    void train(const tensor& x_train, const tensor& y_train);
    float evaluate(const tensor& x, const tensor& y);
    tensor predict(const tensor& x);
};

lstm::lstm(const loss_func &loss, const float lr) {
    this->loss = loss;
    this->lr = lr;

    w_f = glorot_uniform(hidden_size, hidden_size + input_size);
    w_i = glorot_uniform(hidden_size, hidden_size + input_size);
    w_c = glorot_uniform(hidden_size, hidden_size + input_size);
    w_o = glorot_uniform(hidden_size, hidden_size + input_size);
    w_y = glorot_uniform(output_size, hidden_size);

    b_f = zeros({hidden_size, 1});
    b_i = zeros({hidden_size, 1});
    b_c = zeros({hidden_size, 1});
    b_o = zeros({hidden_size, 1});
    b_y = zeros({output_size, 1});

    m_w_f = zeros({hidden_size, hidden_size + input_size});
    m_w_i = zeros({hidden_size, hidden_size + input_size});
    m_w_c = zeros({hidden_size, hidden_size + input_size});
    m_w_o = zeros({hidden_size, hidden_size + input_size});
    m_w_y = zeros({output_size, hidden_size});

    m_b_f = zeros({hidden_size, 1});
    m_b_i = zeros({hidden_size, 1});
    m_b_c = zeros({hidden_size, 1});
    m_b_o = zeros({hidden_size, 1});
    m_b_y = zeros({output_size, 1});

    v_w_f = zeros({hidden_size, hidden_size + input_size});
    v_w_i = zeros({hidden_size, hidden_size + input_size});
    v_w_c = zeros({hidden_size, hidden_size + input_size});
    v_w_o = zeros({hidden_size, hidden_size + input_size});
    v_w_y = zeros({output_size, hidden_size});

    v_b_f = zeros({hidden_size, 1});
    v_b_i = zeros({hidden_size, 1});
    v_b_c = zeros({hidden_size, 1});
    v_b_o = zeros({hidden_size, 1});
    v_b_y = zeros({output_size, 1});
}

void lstm::train(const tensor& x_train, const tensor& y_train) {
    for (auto i = 1; i <= epochs; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        auto [x_sequence, concat_sequence, z_f_sequence, z_i_sequence, i_sequence, z_c_tilde_sequence, c_tilde_sequence, c_sequence, z_o_sequence, o_sequence, h_sequence, y_sequence] = forward(x_train, Phase::TRAIN);

        float error = loss(transpose(y_train), y_sequence.front());

        tensor d_loss_d_h_t_w_f = zeros({batch_size, hidden_size});
        tensor d_loss_d_h_t_w_i = zeros({batch_size, hidden_size});
        tensor d_loss_d_h_t_w_c = zeros({batch_size, hidden_size});
        tensor d_loss_d_h_t_w_o = zeros({batch_size, hidden_size});

        tensor d_loss_d_w_f = zeros({hidden_size, hidden_size + input_size});
        tensor d_loss_d_w_i = zeros({hidden_size, hidden_size + input_size});
        tensor d_loss_d_w_c = zeros({hidden_size, hidden_size + input_size});
        tensor d_loss_d_w_o = zeros({hidden_size, hidden_size + input_size});

        tensor d_loss_d_b_f = zeros({hidden_size, 1});
        tensor d_loss_d_b_i = zeros({hidden_size, 1});
        tensor d_loss_d_b_c = zeros({hidden_size, 1});
        tensor d_loss_d_b_o = zeros({hidden_size, 1});

        float num_samples = static_cast<float>(y_train.shape.front());
        tensor d_loss_d_y = -2.0f / num_samples * (transpose(y_train) - y_sequence.front());

        for (auto j = seq_length; j > 0; --j) {
            if (j == seq_length) {
                tensor d_y_d_h_10 = w_y;

                d_loss_d_h_t_w_f = matmul(transpose(d_loss_d_y), d_y_d_h_10);
                d_loss_d_h_t_w_i = matmul(transpose(d_loss_d_y), d_y_d_h_10);
                d_loss_d_h_t_w_c = matmul(transpose(d_loss_d_y), d_y_d_h_10);
                d_loss_d_h_t_w_o = matmul(transpose(d_loss_d_y), d_y_d_h_10);
            } else {
                d_loss_d_h_t_w_f = matmul(d_loss_d_h_t_w_f * transpose(o_sequence[j] * (1.0f - square(hyperbolic_tangent(c_sequence[j + 1]))) * c_sequence[j + 1] * sigmoid_derivative(z_f_sequence[j])), vslice(w_f, w_f.shape.back() - 1));
                d_loss_d_h_t_w_i = matmul(d_loss_d_h_t_w_i * transpose(o_sequence[j] * (1.0f - square(hyperbolic_tangent(c_sequence[j + 1]))) * c_tilde_sequence[j] * sigmoid_derivative(z_i_sequence[j])), vslice(w_i, w_i.shape.back() - 1));
                d_loss_d_h_t_w_c = matmul(d_loss_d_h_t_w_c * transpose(o_sequence[j] * (1.0f - square(hyperbolic_tangent(c_sequence[j + 1]))) * i_sequence[j] * (1.0f - square(hyperbolic_tangent(z_c_tilde_sequence[j])))), vslice(w_c, w_c.shape.back() - 1));
                d_loss_d_h_t_w_o = matmul(d_loss_d_h_t_w_o * transpose(hyperbolic_tangent(c_sequence[j + 1]) * sigmoid_derivative(z_o_sequence[j])), vslice(w_o, w_o.shape.back() - 1));
            }

            d_loss_d_w_f = d_loss_d_w_f + matmul(transpose(d_loss_d_h_t_w_f) * o_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(c_sequence[j]))) * c_sequence[j] * sigmoid_derivative(z_f_sequence[j - 1]), transpose(concat_sequence[j - 1]));
            d_loss_d_w_i = d_loss_d_w_i + matmul(transpose(d_loss_d_h_t_w_i) * o_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(c_sequence[j]))) * c_tilde_sequence[j - 1] * sigmoid_derivative(z_i_sequence[j - 1]), transpose(concat_sequence[j - 1]));
            d_loss_d_w_c = d_loss_d_w_c + matmul(transpose(d_loss_d_h_t_w_c) * o_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(c_sequence[j]))) * i_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(z_c_tilde_sequence[j - 1]))), transpose(concat_sequence[j - 1]));
            d_loss_d_w_o = d_loss_d_w_o + matmul(transpose(d_loss_d_h_t_w_o) * hyperbolic_tangent(c_sequence[j]) * sigmoid_derivative(z_o_sequence[j - 1]), transpose(concat_sequence[j - 1]));

            d_loss_d_b_f = d_loss_d_b_f + sum(transpose(d_loss_d_h_t_w_f) * o_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(c_sequence[j]))) * c_sequence[j] * sigmoid_derivative(z_f_sequence[j - 1]), 1);
            d_loss_d_b_i = d_loss_d_b_i + sum(transpose(d_loss_d_h_t_w_i) * o_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(c_sequence[j]))) * c_tilde_sequence[j - 1] * sigmoid_derivative(z_i_sequence[j - 1]), 1);
            d_loss_d_b_c = d_loss_d_b_c + sum(transpose(d_loss_d_h_t_w_c) * o_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(c_sequence[j]))) * i_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(z_c_tilde_sequence[j - 1]))), 1);
            d_loss_d_b_o = d_loss_d_b_o + sum(transpose(d_loss_d_h_t_w_o) * hyperbolic_tangent(c_sequence[j]) * sigmoid_derivative(z_o_sequence[j - 1]), 1);
        }

        tensor d_loss_d_w_y  = matmul(d_loss_d_y, transpose(h_sequence.back()));

        t += 1;

        m_w_f = beta1 * m_w_f + (1.0f - beta1) * d_loss_d_w_f;
        m_w_i = beta1 * m_w_i + (1.0f - beta1) * d_loss_d_w_i;
        m_w_c = beta1 * m_w_c + (1.0f - beta1) * d_loss_d_w_c;
        m_w_o = beta1 * m_w_o + (1.0f - beta1) * d_loss_d_w_o;
        m_w_y = beta1 * m_w_y + (1.0f - beta1) * d_loss_d_w_y;

        m_b_f = beta1 * m_b_f + (1.0f - beta1) * d_loss_d_b_f;
        m_b_i = beta1 * m_b_i + (1.0f - beta1) * d_loss_d_b_i;
        m_b_c = beta1 * m_b_c + (1.0f - beta1) * d_loss_d_b_c;
        m_b_o = beta1 * m_b_o + (1.0f - beta1) * d_loss_d_b_o;
        m_b_y = beta1 * m_b_y + (1.0f - beta1) * d_loss_d_y;

        v_w_f = beta2 * v_w_f + (1.0f - beta2) * square(d_loss_d_w_f);
        v_w_i = beta2 * v_w_i + (1.0f - beta2) * square(d_loss_d_w_i);
        v_w_c = beta2 * v_w_c + (1.0f - beta2) * square(d_loss_d_w_c);
        v_w_o = beta2 * v_w_o + (1.0f - beta2) * square(d_loss_d_w_o);
        v_w_y = beta2 * v_w_y + (1.0f - beta2) * square(d_loss_d_w_y);

        v_b_f = beta2 * v_b_f + (1.0f - beta2) * square(d_loss_d_b_f);
        v_b_i = beta2 * v_b_i + (1.0f - beta2) * square(d_loss_d_b_i);
        v_b_c = beta2 * v_b_c + (1.0f - beta2) * square(d_loss_d_b_c);
        v_b_o = beta2 * v_b_o + (1.0f - beta2) * square(d_loss_d_b_o);
        v_b_y = beta2 * v_b_y + (1.0f - beta2) * square(d_loss_d_y);

        tensor m_hat_w_f = m_w_f / (1.0f - powf(beta1, t));
        tensor m_hat_w_i = m_w_i / (1.0f - powf(beta1, t));
        tensor m_hat_w_c = m_w_c / (1.0f - powf(beta1, t));
        tensor m_hat_w_o = m_w_o / (1.0f - powf(beta1, t));
        tensor m_hat_w_y = m_w_y / (1.0f - powf(beta1, t));

        tensor m_hat_b_f = m_b_f / (1.0f - powf(beta1, t));
        tensor m_hat_b_i = m_b_i / (1.0f - powf(beta1, t));
        tensor m_hat_b_c = m_b_c / (1.0f - powf(beta1, t));
        tensor m_hat_b_o = m_b_o / (1.0f - powf(beta1, t));
        tensor m_hat_b_y = m_b_y / (1.0f - powf(beta1, t));

        tensor v_hat_w_f = v_w_f / (1.0f - powf(beta2, t));
        tensor v_hat_w_i = v_w_i / (1.0f - powf(beta2, t));
        tensor v_hat_w_c = v_w_c / (1.0f - powf(beta2, t));
        tensor v_hat_w_o = v_w_o / (1.0f - powf(beta2, t));
        tensor v_hat_w_y = v_w_y / (1.0f - powf(beta2, t));

        tensor v_hat_b_f = v_b_f / (1.0f - powf(beta2, t));
        tensor v_hat_b_i = v_b_i / (1.0f - powf(beta2, t));
        tensor v_hat_b_c = v_b_c / (1.0f - powf(beta2, t));
        tensor v_hat_b_o = v_b_o / (1.0f - powf(beta2, t));
        tensor v_hat_b_y = v_b_y / (1.0f - powf(beta2, t));

        w_f = w_f - lr * m_hat_w_f / (sqrt(v_hat_w_f) + epsilon);
        w_i = w_i - lr * m_hat_w_i / (sqrt(v_hat_w_i) + epsilon);
        w_c = w_c - lr * m_hat_w_c / (sqrt(v_hat_w_c) + epsilon);
        w_o = w_o - lr * m_hat_w_o / (sqrt(v_hat_w_o) + epsilon);
        w_y = w_y - lr * m_hat_w_y / (sqrt(v_hat_w_y) + epsilon);

        b_f = b_f - lr * m_hat_b_f / (sqrt(v_hat_b_f) + epsilon);
        b_i = b_i - lr * m_hat_b_i / (sqrt(v_hat_b_i) + epsilon);
        b_c = b_c - lr * m_hat_b_c / (sqrt(v_hat_b_c) + epsilon);
        b_o = b_o - lr * m_hat_b_o / (sqrt(v_hat_b_o) + epsilon);
        b_y = b_y - lr * m_hat_b_y / (sqrt(v_hat_b_y) + epsilon);

        // w_f = w_f - lr * d_loss_d_w_f;
        // w_i = w_i - lr * d_loss_d_w_i;
        // w_c = w_c - lr * d_loss_d_w_c;
        // w_o = w_o - lr * d_loss_d_w_o;
        // w_y = w_y - lr * d_loss_d_w_y;

        // b_f = b_f - lr * d_loss_d_b_f;
        // b_i = b_i - lr * d_loss_d_b_i;
        // b_c = b_c - lr * d_loss_d_b_c;
        // b_o = b_o - lr * d_loss_d_b_o;
        // b_y = b_y - lr * d_loss_d_y;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remaining_ms = duration - seconds;

        std::cout << "Epoch " << i << "/" << epochs << std::endl << seconds.count() << "s " << remaining_ms.count() << "ms/step - loss: " << error << std::endl;
    }
}

float lstm::evaluate(const tensor& x, const tensor& y) {
    auto [x_sequence, concat_sequence, z_f_sequence, z_i_sequence, i_sequence, z_c_tilde_sequence, c_tilde_sequence, c_sequence, z_o_sequence, o_sequence, h_sequence, y_sequence] = forward(x, Phase::TEST);
    return loss(transpose(y), y_sequence.front());
}

tensor lstm::predict(const tensor& x) {
    auto [x_sequence, concat_sequence, z_f_sequence, z_i_sequence, i_sequence, z_c_tilde_sequence, c_tilde_sequence, c_sequence, z_o_sequence, o_sequence, h_sequence, y_sequence] = forward(x, Phase::TEST);
    return transpose(y_sequence.front());
}

std::array<std::vector<tensor>, 12> lstm::forward(const tensor& x, enum Phase phase) {
    std::vector<tensor> x_sequence;
    std::vector<tensor> concat_sequence;
    std::vector<tensor> z_f_sequence;
    std::vector<tensor> z_i_sequence;
    std::vector<tensor> i_sequence;
    std::vector<tensor> z_c_tilde_sequence;
    std::vector<tensor> c_tilde_sequence;
    std::vector<tensor> c_sequence;
    std::vector<tensor> z_o_sequence;
    std::vector<tensor> o_sequence;
    std::vector<tensor> h_sequence;
    std::vector<tensor> y_sequence;

    if (phase == Phase::TRAIN)
        batch_size = 8317;
    else
        batch_size = 2072;

    tensor c_t = zeros({hidden_size, batch_size});
    c_sequence.push_back(c_t);

    tensor h_t = zeros({hidden_size, batch_size});
    h_sequence.push_back(h_t);

    for (auto i = 0; i < seq_length; ++i) {
        size_t idx = i;

        tensor x_t = zeros({batch_size, input_size});

        for (auto j = 0; j < batch_size; ++j) {
            x_t[j] = x[idx];
            idx += seq_length;
        }

        tensor concat_t = vstack({h_t, transpose(x_t)});

        tensor z_f_t = matmul(w_f, concat_t) + b_f;
        tensor f_t = sigmoid(z_f_t);

        tensor z_i_t = matmul(w_i, concat_t) + b_i;
        tensor i_t = sigmoid(z_i_t);

        tensor z_c_tilde_t = matmul(w_c, concat_t) + b_c;
        tensor c_tilde_t = hyperbolic_tangent(z_c_tilde_t);

        c_t = f_t * c_t + i_t * c_tilde_t;

        tensor z_o_t = matmul(w_o, concat_t) + b_o;
        tensor o_t = sigmoid(z_o_t);

        h_t = o_t * hyperbolic_tangent(c_t);

        tensor y_t = matmul(w_y, h_t) + b_y;

        x_sequence.push_back(x_t);
        concat_sequence.push_back(concat_t);
        z_f_sequence.push_back(z_f_t);
        z_i_sequence.push_back(z_i_t);
        i_sequence.push_back(i_t);
        z_c_tilde_sequence.push_back(z_c_tilde_t);
        c_tilde_sequence.push_back(c_tilde_t);
        c_sequence.push_back(c_t);
        z_o_sequence.push_back(z_o_t);
        o_sequence.push_back(o_t);
        h_sequence.push_back(h_t);

        if (i == seq_length - 1)
            y_sequence.push_back(y_t);
    }

    std::array<std::vector<tensor>, 12> sequences;

    sequences[0]  = x_sequence;
    sequences[1]  = concat_sequence;
    sequences[2]  = z_f_sequence;
    sequences[3]  = z_i_sequence;
    sequences[4]  = i_sequence;
    sequences[5]  = z_c_tilde_sequence;
    sequences[6]  = c_tilde_sequence;
    sequences[7]  = c_sequence;
    sequences[8]  = z_o_sequence;
    sequences[9]  = o_sequence;
    sequences[10]  = h_sequence;
    sequences[11] = y_sequence;

    return sequences;
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

    lstm model = lstm(mean_squared_error, 0.01f);
    model.train(x_y_train.first, x_y_train.second);

    auto test_loss = model.evaluate(x_y_test.first, x_y_test.second);
    auto predict = scaler.inverse_transform(model.predict(x_y_test.first));

    x_y_test.second = scaler.inverse_transform(x_y_test.second);

    for (auto i = 0; i < x_y_test.second.size; ++i)
        std::cout << x_y_test.second[i] << " " << predict[i] << std::endl;

    std::cout << "Test  loss: " << test_loss << std::endl;

    return 0;
}