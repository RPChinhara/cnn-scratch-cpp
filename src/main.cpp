#include "acts.h"
#include "arrs.h"
#include "datas.h"
#include "linalg.h"
#include "losses.h"
#include "lyrs.h"
#include "math.hpp"
#include "preproc.h"
#include "rd.h"
#include "tensor.h"

#include <chrono>

class gru2 {
  private:
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

    tensor w_z;
    tensor w_r;
    tensor w_h;
    tensor w_y;

    tensor b_z;
    tensor b_r;
    tensor b_h;
    tensor b_y;

    tensor m_w_z;
    tensor m_w_r;
    tensor m_w_h;
    tensor m_w_y;

    tensor m_b_z;
    tensor m_b_r;
    tensor m_b_h;
    tensor m_b_y;

    tensor v_w_z;
    tensor v_w_r;
    tensor v_w_h;
    tensor v_w_y;

    tensor v_b_z;
    tensor v_b_r;
    tensor v_b_h;
    tensor v_b_y;

    enum Phase {
      TRAIN,
      TEST
    };

    std::array<std::vector<tensor>, 11> forward(const tensor &x, enum Phase phase);

  public:
    gru2(const float lr);
    void train(const tensor &x_train, const tensor &y_train);
    float evaluate(const tensor &x, const tensor &y);
    tensor predict(const tensor &x);
};

gru2::gru2(const float lr) {
    this->lr = lr;

    w_z = glorot_uniform(hidden_size, hidden_size + input_size);
    w_r = glorot_uniform(hidden_size, hidden_size + input_size);
    w_h = glorot_uniform(hidden_size, hidden_size + input_size);
    w_y = glorot_uniform(output_size, hidden_size);

    b_z = zeros({hidden_size, 1});
    b_r = zeros({hidden_size, 1});
    b_h = zeros({hidden_size, 1});
    b_y = zeros({output_size, 1});

    m_w_z = zeros({hidden_size, hidden_size + input_size});
    m_w_r = zeros({hidden_size, hidden_size + input_size});
    m_w_h = zeros({hidden_size, hidden_size + input_size});
    m_w_y = zeros({output_size, hidden_size});

    m_b_z = zeros({hidden_size, 1});
    m_b_r = zeros({hidden_size, 1});
    m_b_h = zeros({hidden_size, 1});
    m_b_y = zeros({output_size, 1});

    v_w_z = zeros({hidden_size, hidden_size + input_size});
    v_w_r = zeros({hidden_size, hidden_size + input_size});
    v_w_h = zeros({hidden_size, hidden_size + input_size});
    v_w_y = zeros({output_size, hidden_size});

    v_b_z = zeros({hidden_size, 1});
    v_b_r = zeros({hidden_size, 1});
    v_b_h = zeros({hidden_size, 1});
    v_b_y = zeros({output_size, 1});
}

void gru2::train(const tensor &x_train, const tensor &y_train) {
    for (auto i = 1; i <= epochs; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        auto [x_sequence, concat_sequence, z_t_z_sequence, z_sequence, r_t_z_sequence, r_sequence, concat_2_sequence, h_hat_t_z_sequence, h_hat_t_sequence, h_sequence, y_sequence] = forward(x_train, Phase::TRAIN);

        float error = mean_squared_error(transpose(y_train), y_sequence.front());

        tensor d_loss_d_h_t_w_z = zeros({batch_size, hidden_size});
        tensor d_loss_d_h_t_w_r = zeros({batch_size, hidden_size});
        tensor d_loss_d_h_t_w_h = zeros({batch_size, hidden_size});

        tensor d_loss_d_w_z = zeros({hidden_size, hidden_size + input_size});
        tensor d_loss_d_w_r = zeros({hidden_size, hidden_size + input_size});
        tensor d_loss_d_w_h = zeros({hidden_size, hidden_size + input_size});

        tensor d_loss_d_b_z = zeros({hidden_size, 1});
        tensor d_loss_d_b_r = zeros({hidden_size, 1});
        tensor d_loss_d_b_h = zeros({hidden_size, 1});

        float num_samples = static_cast<float>(y_train.shape.front());
        tensor d_loss_d_y = -2.0f / num_samples * (transpose(y_train) - y_sequence.front());

        // d_loss_d_h_t_w_h:   (8317, 50)
        // z_t_sequence:       (50, 8317)
        // concat_2_sequence:  (51, 8317)
        // h_hat_t_z_sequence: (50, 8317)
        // only h_sequence is of size 11, other is 10

        // (50, 51) * (51, 8317)

        // 1 2 3     1 2 3 1 2 3
        // 1 2 3     1 2 3 1 2 3
        // 1 2 3     1 2 3 1 2 3

        for (auto j = seq_length; j > 0; --j) {
            if (j == seq_length) {
                tensor d_y_d_h_10 = w_y;

                d_loss_d_h_t_w_z = matmul(transpose(d_loss_d_y), d_y_d_h_10);
                d_loss_d_h_t_w_r = matmul(transpose(d_loss_d_y), d_y_d_h_10);
                d_loss_d_h_t_w_h = matmul(transpose(d_loss_d_y), d_y_d_h_10);
            } else {
                d_loss_d_h_t_w_z = matmul(d_loss_d_h_t_w_z * transpose(h_hat_t_sequence[j] * sigmoid_derivative(z_t_z_sequence[j])), vslice(w_z, w_z.shape.back() - 1));
                d_loss_d_h_t_w_r = matmul(d_loss_d_h_t_w_r * transpose(z_sequence[j] * (1.0f - square(hyperbolic_tangent(h_hat_t_z_sequence[j]))) * matmul(vslice(w_h, w_h.shape.back() - 1), h_sequence[j + 1]) * sigmoid_derivative(r_t_z_sequence[j])), vslice(w_r, w_r.shape.back() - 1));
                d_loss_d_h_t_w_h = d_loss_d_h_t_w_h * transpose(z_sequence[j] * (1.0f - square(hyperbolic_tangent(h_hat_t_z_sequence[j]))) * matmul(vslice(w_h, w_h.shape.back() - 1), r_sequence[j]));
            }
            // NOTE: Could be (1.0f - square(h_hat_t_sequence[j - 1])), and this applies to all that have partial derivative of tanh
            d_loss_d_w_z = d_loss_d_w_z + matmul(transpose(d_loss_d_h_t_w_z) * h_hat_t_sequence[j - 1] * sigmoid_derivative(z_t_z_sequence[j - 1]), transpose(concat_sequence[j - 1]));
            d_loss_d_w_r = d_loss_d_w_r + matmul(transpose(d_loss_d_h_t_w_r) * z_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(h_hat_t_z_sequence[j - 1]))) * matmul(vslice(w_h, w_h.shape.back() - 1), h_sequence[j]) * sigmoid_derivative(r_t_z_sequence[j - 1]), transpose(concat_sequence[j - 1]));
            d_loss_d_w_h = d_loss_d_w_h + matmul(transpose(d_loss_d_h_t_w_h) * z_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(h_hat_t_z_sequence[j - 1]))), transpose(concat_2_sequence[j - 1]));

            d_loss_d_b_z = d_loss_d_b_z + sum(transpose(d_loss_d_h_t_w_z) * h_hat_t_sequence[j - 1] * sigmoid_derivative(z_t_z_sequence[j - 1]), 1);
            d_loss_d_b_r = d_loss_d_b_r + sum(transpose(d_loss_d_h_t_w_r) * z_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(h_hat_t_z_sequence[j - 1]))) * matmul(vslice(w_h, w_h.shape.back() - 1), h_sequence[j]) * sigmoid_derivative(r_t_z_sequence[j - 1]), 1);
            d_loss_d_b_h = d_loss_d_b_h + sum(transpose(d_loss_d_h_t_w_h) * z_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(h_hat_t_z_sequence[j - 1]))), 1);
        }

        // OBSERVE:
        // d_loss_d_h_t_w_h = matmul(d_loss_d_h_t_w_h * transpose(z_sequence[j] * (1.0f - square(hyperbolic_tangent(h_hat_t_z_sequence[j])))), vslice(w_h, w_h.shape.back() - 1))
        // test losses: 0.000609079, 0.0015165, 0.000932636, 0.00047658, 0.00147116, 0.000575935


        // d_loss_d_w_r = d_loss_d_w_r + matmul(transpose(d_loss_d_h_t_w_r) * z_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(h_hat_t_z_sequence[j - 1]))) * matmul(vslice(w_h, w_h.shape.back() - 1), h_sequence[j]) * sigmoid_derivative(r_t_z_sequence[j - 1]), transpose(concat_sequence[j - 1]));
        // test losses: 0.000374682, 0.000617785, 0.000426879, 9.53731e-05, 0.00111133, 7.91521e-05

        // d_loss_d_h_t_w_h = d_loss_d_h_t_w_h * transpose(z_sequence[j] * (1.0f - square(hyperbolic_tangent(h_hat_t_z_sequence[j]))) * matmul(vslice(w_h, w_h.shape.back() - 1), r_sequence[j]));
        // test losses: 0.000197716


        // tensor concat_t = vstack({h_t, transpose(x_t)});

        // tensor z_t_z = matmul(w_z, concat_t) + b_z;
        // tensor z_t = sigmoid(z_t_z);

        // tensor r_t_z = matmul(w_r, concat_t) + b_r;
        // tensor r_t = sigmoid(r_t_z);

        // tensor concat_2_t = vstack({r_t * h_t, transpose(x_t)});

        // tensor h_hat_t_z = matmul(w_h, concat_2_t) + b_h;
        // tensor h_hat_t = hyperbolic_tangent(h_hat_t_z);

        // h_t = (ones - z_t) * h_t + z_t * h_hat_t;

        // tensor y_t_z = matmul(w_y, h_t) + b_y;
        // tensor y_t = softmax(y_t_z);

        // =====================================================================================================================================
        // dL/dw_h: (dL/dy * dy/dh_10) * dh_10/dh_hat_10 * dh_hat_10/dw_h
        //          (dL/dy * dy/dh_10) * dh_10/dh_hat_10 * dh_hat_10/dh_9) * dh_9/dh_hat_9 * dh_hat_9/dw_h

        // dL/dw_r: (dL/dy * dy/dh_10) * dh_10/dh_hat_10 * dh_hat_10/dr_10 * dr_10/dw_r
        //          (dL/dy * dy/dh_10 * dh_10/dh_hat_10 * dh_hat_10/dr_10 * dr_10/dh_9) * dh_9/dh_hat_9 * dh_hat_9/dr_9 * dr_9/dw_r

        // dL/dw_z: (dL/dy * dy/dh_10) * dh_10/dz_10 * dz_10/dw_z
        //          (dL/dy * dy/dh_10 * dh_10/dz_10 * dz_10/dh_9) * dh_9/dz_9 * dz_9/dw_z
        // =====================================================================================================================================

        tensor d_loss_d_w_y  = matmul(d_loss_d_y, transpose(h_sequence.back()));

        t += 1;

        m_w_z = beta1 * m_w_z + (1.0f - beta1) * d_loss_d_w_z;
        m_w_r = beta1 * m_w_r + (1.0f - beta1) * d_loss_d_w_r;
        m_w_h = beta1 * m_w_h + (1.0f - beta1) * d_loss_d_w_h;
        m_w_y = beta1 * m_w_y + (1.0f - beta1) * d_loss_d_w_y;

        m_b_z = beta1 * m_b_z + (1.0f - beta1) * d_loss_d_b_z;
        m_b_r = beta1 * m_b_r + (1.0f - beta1) * d_loss_d_b_r;
        m_b_h = beta1 * m_b_h + (1.0f - beta1) * d_loss_d_b_h;
        m_b_y = beta1 * m_b_y + (1.0f - beta1) * d_loss_d_y;

        v_w_z = beta2 * v_w_z + (1.0f - beta2) * square(d_loss_d_w_z);
        v_w_r = beta2 * v_w_r + (1.0f - beta2) * square(d_loss_d_w_r);
        v_w_h = beta2 * v_w_h + (1.0f - beta2) * square(d_loss_d_w_h);
        v_w_y = beta2 * v_w_y + (1.0f - beta2) * square(d_loss_d_w_y);

        v_b_z = beta2 * v_b_z + (1.0f - beta2) * square(d_loss_d_b_z);
        v_b_r = beta2 * v_b_r + (1.0f - beta2) * square(d_loss_d_b_r);
        v_b_h = beta2 * v_b_h + (1.0f - beta2) * square(d_loss_d_b_h);
        v_b_y = beta2 * v_b_y + (1.0f - beta2) * square(d_loss_d_y);

        tensor m_hat_w_z = m_w_z / (1.0f - powf(beta1, t));
        tensor m_hat_w_r = m_w_r / (1.0f - powf(beta1, t));
        tensor m_hat_w_h = m_w_h / (1.0f - powf(beta1, t));
        tensor m_hat_w_y = m_w_y / (1.0f - powf(beta1, t));

        tensor m_hat_b_z = m_b_z / (1.0f - powf(beta1, t));
        tensor m_hat_b_r = m_b_r / (1.0f - powf(beta1, t));
        tensor m_hat_b_h = m_b_h / (1.0f - powf(beta1, t));
        tensor m_hat_b_y = m_b_y / (1.0f - powf(beta1, t));

        tensor v_hat_w_z = v_w_z / (1.0f - powf(beta2, t));
        tensor v_hat_w_r = v_w_r / (1.0f - powf(beta2, t));
        tensor v_hat_w_h = v_w_h / (1.0f - powf(beta2, t));
        tensor v_hat_w_y = v_w_y / (1.0f - powf(beta2, t));

        tensor v_hat_b_z = v_b_z / (1.0f - powf(beta2, t));
        tensor v_hat_b_r = v_b_r / (1.0f - powf(beta2, t));
        tensor v_hat_b_h = v_b_h / (1.0f - powf(beta2, t));
        tensor v_hat_b_y = v_b_y / (1.0f - powf(beta2, t));

        w_z = w_z - lr * m_hat_w_z / (sqrt(v_hat_w_z) + epsilon);
        w_r = w_r - lr * m_hat_w_r / (sqrt(v_hat_w_r) + epsilon);
        w_h = w_h - lr * m_hat_w_h / (sqrt(v_hat_w_h) + epsilon);
        w_y = w_y - lr * m_hat_w_y / (sqrt(v_hat_w_y) + epsilon);

        b_z = b_z - lr * m_hat_b_z / (sqrt(v_hat_b_z) + epsilon);
        b_r = b_r - lr * m_hat_b_r / (sqrt(v_hat_b_r) + epsilon);
        b_h = b_h - lr * m_hat_b_h / (sqrt(v_hat_b_h) + epsilon);
        b_y = b_y - lr * m_hat_b_y / (sqrt(v_hat_b_y) + epsilon);

        // w_z = w_z - lr * d_loss_d_w_z;
        // w_r = w_r - lr * d_loss_d_w_r;
        // w_h = w_h - lr * d_loss_d_w_h;
        // w_y = w_y - lr * d_loss_d_w_y;

        // b_z = b_z - lr * d_loss_d_b_z;
        // b_r = b_r - lr * d_loss_d_b_r;
        // b_h = b_h - lr * d_loss_d_b_h;
        // b_y = b_y - lr * d_loss_d_y;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remaining_ms = duration - seconds;

        std::cout << "Epoch " << i << "/" << epochs << std::endl << seconds.count() << "s " << remaining_ms.count() << "ms/step - loss: " << error << std::endl;
    }
}

float gru2::evaluate(const tensor &x, const tensor &y) {
    auto [x_sequence, concat_sequence, z_t_z_sequence, z_sequence, r_t_z_sequence, r_sequence, concat_2_sequence, h_hat_t_z_sequence, h_hat_t_sequence, h_sequence, y_sequence] = forward(x, Phase::TEST);
    return mean_squared_error(transpose(y), y_sequence.front());
}

tensor gru2::predict(const tensor &x) {
    auto [x_sequence, concat_sequence, z_t_z_sequence, z_sequence, r_t_z_sequence, r_sequence, concat_2_sequence, h_hat_t_z_sequence, h_hat_t_sequence, h_sequence, y_sequence] = forward(x, Phase::TEST);
    return transpose(y_sequence.front());
}

std::array<std::vector<tensor>, 11> gru2::forward(const tensor &x, enum Phase phase) {
    std::vector<tensor> x_sequence;
    std::vector<tensor> concat_sequence;
    std::vector<tensor> z_t_z_sequence;
    std::vector<tensor> z_sequence;
    std::vector<tensor> r_t_z_sequence;
    std::vector<tensor> r_sequence;
    std::vector<tensor> concat_2_sequence;
    std::vector<tensor> h_hat_t_z_sequence;
    std::vector<tensor> h_hat_t_sequence;
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

        tensor concat_t = vstack({h_t, transpose(x_t)});

        tensor z_t_z = matmul(w_z, concat_t) + b_z;
        tensor z_t = sigmoid(z_t_z);

        tensor r_t_z = matmul(w_r, concat_t) + b_r;
        tensor r_t = sigmoid(r_t_z);

        tensor concat_2_t = vstack({r_t * h_t, transpose(x_t)});

        tensor h_hat_t_z = matmul(w_h, concat_2_t) + b_h;
        tensor h_hat_t = hyperbolic_tangent(h_hat_t_z);

        h_t = (1.0f - z_t) * h_t + z_t * h_hat_t;

        tensor y_t = matmul(w_y, h_t) + b_y;

        x_sequence.push_back(x_t);
        concat_sequence.push_back(concat_t);
        z_t_z_sequence.push_back(z_t_z);
        z_sequence.push_back(z_t);
        r_t_z_sequence.push_back(r_t_z);
        r_sequence.push_back(r_t);
        concat_2_sequence.push_back(concat_2_t);
        h_hat_t_z_sequence.push_back(h_hat_t_z);
        h_hat_t_sequence.push_back(h_hat_t);
        h_sequence.push_back(h_t);

        if (i == seq_length - 1)
            y_sequence.push_back(y_t);
    }

    std::array<std::vector<tensor>, 11> sequences;

    sequences[0]  = x_sequence;
    sequences[1]  = concat_sequence;
    sequences[2]  = z_t_z_sequence;
    sequences[3]  = z_sequence;
    sequences[4]  = r_t_z_sequence;
    sequences[5]  = r_sequence;
    sequences[6]  = concat_2_sequence;
    sequences[7]  = h_hat_t_z_sequence;
    sequences[8]  = h_hat_t_sequence;
    sequences[9]  = h_sequence;
    sequences[10] = y_sequence;

    return sequences;
}

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

int main() {
    tensor data = load_aapl();

    min_max_scaler2 scaler;
    scaler.fit(data);
    tensor scaled_data = scaler.transform(data);

    auto train_test = split(scaled_data, 0.2f);

    auto x_y_train = create_sequences(train_test.first, 10);
    auto x_y_test = create_sequences(train_test.second, 10);

    gru2 model = gru2(0.01f);
    model.train(x_y_train.first, x_y_train.second);

    auto test_loss = model.evaluate(x_y_test.first, x_y_test.second);
    auto predict = scaler.inverse_transform(model.predict(x_y_test.first));

    x_y_test.second = scaler.inverse_transform(x_y_test.second);

    for (auto i = 0; i < x_y_test.second.size; ++i)
        std::cout << x_y_test.second[i] << " " << predict[i] << std::endl;

    std::cout << "Test  loss: " << test_loss << std::endl;

    return 0;
}