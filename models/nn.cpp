#include "activations.h"
#include "arrays.h"
#include "datasets.h"
#include "derivatives.h"
#include "initializers.h"
#include "linalg.h"
#include "losses.h"
#include "mathematics.h"
#include "metrics.h"
#include "preprocessing.h"
#include "random.h"
#include "regularizations.h"

#include <array>
#include <random>

#define EARLY_STOPPING_ENABLED          1
#define GRADIENT_CLIPPING_ENABLED       0
#define LEARNING_RATE_SCHEDULER_ENABLED 1
#define L1_REGULARIZATION_ENABLED       0
#define L2_REGULARIZATION_ENABLED       0
#define L1L2_REGULARIZATION_ENABLED     0
#define MOMENTUM_ENABLED                0

constexpr auto ACCURACY                                  = &categorical_accuracy;
constexpr unsigned short BATCH_SIZE                      = 8;
constexpr unsigned short EPOCHS                          = 100;
[[maybe_unused]] constexpr float GRADIENT_CLIP_THRESHOLD = 8.0f;
constexpr std::array<unsigned short, 4> LAYERS           = { 4, 32, 32, 3 };
float LEARNING_RATE                                      = 0.01f;
constexpr auto LOSS                                      = &categorical_crossentropy;
[[maybe_unused]] constexpr float L1_LAMBDA               = 0.05f;
[[maybe_unused]] constexpr float L2_LAMBDA               = 0.06f;
[[maybe_unused]] constexpr float MOMENTUM                = 0.1f;
[[maybe_unused]] constexpr unsigned short PATIENCE       = 12;

using TensorArray = std::array<Tensor, LAYERS.size() - 1>;

TensorArray forward_propagation(const Tensor& input, const TensorArray& w, const TensorArray& b) {
    TensorArray z;
    TensorArray a;

    for (unsigned short i = 0; i < LAYERS.size() - 1; ++i) {
        if (i == 0) {
            z[i] = (matmul(input, w[i]) + b[i]);
            a[i] = (relu(z[i]));
        } else {
            z[i] = (matmul(a[i - 1], w[i]) + b[i]);
            if (i == 1)
                a[i] = (relu(z[i]));
            else
                a[i] = (softmax(z[i]));
        }
    }

    return a;
}

std::pair<TensorArray, TensorArray> init_parameters() {
    TensorArray w;
    TensorArray b;

    for (unsigned int i = 0; i < LAYERS.size() - 1; ++i) {
        w[i] = normal_distribution({ LAYERS[i], LAYERS[i + 1] }, 0.0f, 2.0f);
        b[i] = zeros({ 1, LAYERS[i + 1] });
    }

    return std::make_pair(w, b);
}

void log_metrics(const std::string& data, const Tensor& y_true, const Tensor& y_pred, const TensorArray *w = nullptr) {
    if (data != "test") {
        if (!w) {
            std::cout << " - " << data << " loss: " << LOSS(y_true, y_pred) << " - " << data << " accuracy: " << ACCURACY(y_true, y_pred);
        } else {
            #if L1_REGULARIZATION_ENABLED && !L2_REGULARIZATION_ENABLED && !L1L2_REGULARIZATION_ENABLED
                // TODO: Adding l1 is manual now meaning it can't adapt to any number of hidden layers. I have to fix this.
                std::cout << " - " << data << " loss: " << LOSS(y_true, y_pred) + l1(L1_LAMBDA, (*w)[0]) + l1(L1_LAMBDA, (*w)[1]) + l1(L1_LAMBDA, (*w)[2]) << " - " << data << " accuracy: " << ACCURACY(y_true, y_pred);
            #elif L2_REGULARIZATION_ENABLED && !L1_REGULARIZATION_ENABLED && !L1L2_REGULARIZATION_ENABLED
                std::cout << " - " << data << " loss: " << LOSS(y_true, y_pred) + l2(L2_LAMBDA, (*w)[0]) + l2(L2_LAMBDA, (*w)[1]) + l2(L2_LAMBDA, (*w)[2]) << " - " << data << " accuracy: " << ACCURACY(y_true, y_pred);
            #elif L1L2_REGULARIZATION_ENABLED && !L1_REGULARIZATION_ENABLED && !L2_REGULARIZATION_ENABLED
                std::cout << " - " << data << " loss: " << LOSS(y_true, y_pred) + l1(L1_LAMBDA, (*w)[0]) + l1(L1_LAMBDA, (*w)[1]) + l1(L1_LAMBDA, (*w)[2]) + l2(L2_LAMBDA, (*w)[0]) + l2(L2_LAMBDA, (*w)[1]) + l2(L2_LAMBDA, (*w)[2]) << " - " << data << " accuracy: " << ACCURACY(y_true, y_pred);
            #else
                std::cerr << "\nChoose only one regularization." << std::endl;
                exit(1);
            #endif
        }
    } else {
        std::cout << data << " loss: " << LOSS(y_true, y_pred) << " - " << data << " accuracy: " << ACCURACY(y_true, y_pred);
    }
}

int main() { 
    Iris iris            = load_iris();
    Tensor x             = iris.features;
    Tensor y             = iris.target;
    y                    = one_hot(y, 3);
    TrainTest train_temp = train_test_split(x, y, 0.2, 42);
    TrainTest val_test   = train_test_split(train_temp.x_second, train_temp.y_second, 0.5, 42);
    train_temp.x_first   = min_max_scaler(train_temp.x_first);
    val_test.x_first     = min_max_scaler(val_test.x_first);
    val_test.x_second    = min_max_scaler(val_test.x_second);

    // TODO: Maybe use w_b.first and w_b.second directly rather than copy into w and b?

    auto w_b = init_parameters();
    TensorArray w = w_b.first;
    TensorArray b = w_b.second;

    #if MOMENTUM_ENABLED
        auto w_b_m = init_parameters();
        TensorArray w_m = w_b_m.first;
        TensorArray b_m = w_b_m.second;
    #endif

    // TODO: Try batch normalization again (I saw it was used in SOTA model in a paper so I might need to work on this).
    // TODO: Use cross-validation technique?
    for (unsigned short i = 1; i <= EPOCHS; ++i) {
        #if LEARNING_RATE_SCHEDULER_ENABLED
            if (i > 10 && i < 20)
                LEARNING_RATE = 0.009f;
            else if (i > 20 && i < 30)
                LEARNING_RATE = 0.005f;
            else
                LEARNING_RATE = 0.001f;
        #endif

        std::random_device rd;
        auto rd_num = rd();
        Tensor x_shuffled = shuffle(train_temp.x_first, rd_num);
        Tensor y_shuffled = shuffle(train_temp.y_first, rd_num);

        Tensor y_batch;
        TensorArray  a;

        // TODO: For loop used for mimi batch gradient process multiple examples in parallel utilizing GPUs. That's the main reason facilitating mini-batch training (use std::thread).
        for (unsigned int j = 0; j < train_temp.x_first._shape.front(); j += BATCH_SIZE) {
            Tensor x_batch = slice(x_shuffled, j, j + BATCH_SIZE);
            y_batch        = slice(y_shuffled, j, j + BATCH_SIZE);

            // regularizer::dropout(0.1f, x);
            a = forward_propagation(x_batch, w, b);

            // TODO: Implement Adam and AdamW.

            std::vector<Tensor> dl_dz, dl_dw, dl_db;

            for (unsigned short i = LAYERS.size() - 1; 0 < i; --i) {
                if (i == LAYERS.size() - 1)
                    dl_dz.push_back(categorical_crossentropy_prime(y_batch, a.back()));
                else
                    dl_dz.push_back(matmul(dl_dz[(LAYERS.size() - 2) - i], w[i].T()) * relu_prime(a[i - 1]));

                // TODO: I could use above '(LAYERS.size() - 2) - i' so that I don't have to use idx, and this applies to other functions use idx. 
            }

            for (unsigned short i = LAYERS.size() - 1; 0 < i; --i) {
                if (i == 1)
                    dl_dw.push_back(matmul(x_batch.T(), dl_dz[(LAYERS.size() - 1) - i]));
                else
                    dl_dw.push_back(matmul(a[i - 2].T(), dl_dz[(LAYERS.size() - 1) - i]));
            }

            for (unsigned short i = 0; i < LAYERS.size() - 1; ++i) {
                dl_db.push_back(sum(dl_dz[i], 0));
            }

            #if GRADIENT_CLIPPING_ENABLED
                for (unsigned short i = 0; i < LAYERS.size() - 1; ++i) {
                    dl_dw[i] = clip_by_value(dl_dw[i], -GRADIENT_CLIP_THRESHOLD, GRADIENT_CLIP_THRESHOLD);
                    dl_db[i] = clip_by_value(dl_db[i], -GRADIENT_CLIP_THRESHOLD, GRADIENT_CLIP_THRESHOLD);
                }
            #endif

            for (short i = LAYERS.size() - 2; 0 <= i; --i) {
                w[i] -= LEARNING_RATE * dl_dw[(LAYERS.size() - 2) - i];
                b[i] -= LEARNING_RATE * dl_db[(LAYERS.size() - 2) - i];
            }

           /* Tensor dl_dz3 = categorical_crossentropy_prime(y_batch, a.back()); // dl/dz3 = dl/dy dy/dz3
            Tensor dl_dz2 = matmul(dl_dz3, w[2].T()) * relu_prime(a[1]); // dl/dz2 = dl_dz3 dz3/da2 da2/z2
            Tensor dl_dz1 = matmul(dl_dz2, w[1].T()) * relu_prime(a[0]); // dl/dz1 = dl_dz2 dz2/da1 da1/z1

            #if L1_REGULARIZATION_ENABLED && !L2_REGULARIZATION_ENABLED && !L1L2_REGULARIZATION_ENABLED
                Tensor dl_dw3 = matmul(a[1].T(), dl_dz3) + l1_prime(L1_LAMBDA, w.back()); // dl/dw3 = d/w3 (categorical_crossentropy) + d/w3 (l1) = dl_dz3 dz3/dw3 + dl1/w3
                Tensor dl_dw2 = matmul(a[0].T(), dl_dz2) + l1_prime(L1_LAMBDA, w[1]); // dl/dw2 = d/w2 (categorical_crossentropy) + d/w2 (l1) = dl_dz2 dz2/dw2 + dl1/w2
                Tensor dl_dw1 = matmul(x_batch.T(), dl_dz1) + l1_prime(L1_LAMBDA, w[0]); // dl/dw1 = d/w1 (categorical_crossentropy) + d/w1 (l1) = dl_dz1 dz1/dw1 + dl1/w1
            #elif L2_REGULARIZATION_ENABLED && !L1_REGULARIZATION_ENABLED && !L1L2_REGULARIZATION_ENABLED
                Tensor dl_dw3 = matmul(a[1].T(), dl_dz3) + l2_prime(L2_LAMBDA, w.back()); // dl/dw3 = d/w3 (categorical_crossentropy) + d/w3 (l1) = dl_dz3 dz3/dw3 + dl1/w3
                Tensor dl_dw2 = matmul(a[0].T(), dl_dz2) + l2_prime(L2_LAMBDA, w[1]); // dl/dw2 = d/w2 (categorical_crossentropy) + d/w2 (l1) = dl_dz2 dz2/dw2 + dl1/w2
                Tensor dl_dw1 = matmul(x_batch.T(), dl_dz1) + l2_prime(L2_LAMBDA, w[0]); // dl/dw1 = d/w1 (categorical_crossentropy) + d/w1 (l1) = dl_dz1 dz1/dw1 + dl1/w1
            #elif L1L2_REGULARIZATION_ENABLED && !L1_REGULARIZATION_ENABLED && !L2_REGULARIZATION_ENABLED
                Tensor dl_dw3 = matmul(a[1].T(), dl_dz3) + l1_prime(L1_LAMBDA, w.back()) + l2_prime(L2_LAMBDA, w.back()); // dl/dw3 = d/w3 (categorical_crossentropy) + d/w3 (l1) = dl_dz3 dz3/dw3 + dl1/w3
                Tensor dl_dw2 = matmul(a[0].T(), dl_dz2) + l1_prime(L1_LAMBDA, w[1]) + l2_prime(L2_LAMBDA, w[1]); // dl/dw2 = d/w2 (categorical_crossentropy) + d/w2 (l1) = dl_dz2 dz2/dw2 + dl1/w2
                Tensor dl_dw1 = matmul(x_batch.T(), dl_dz1) + l1_prime(L1_LAMBDA, w[0]) + l2_prime(L2_LAMBDA, w[0]); // dl/dw1 = d/w1 (categorical_crossentropy) + d/w1 (l1) = dl_dz1 dz1/dw1 + dl1/w1
            #else
                Tensor dl_dw3 = matmul(a[1].T(), dl_dz3); // dl/dw3 = dl_dz3 dz3/dw3
                Tensor dl_dw2 = matmul(a[0].T(), dl_dz2); // dl/dw2 = dl_dz2 dz2/dw2
                Tensor dl_dw1 = matmul(x_batch.T(), dl_dz1); // dl/dw1 = dl_dz1 dz1/dw1
            #endif

            Tensor dl_db3 = sum(dl_dz3, 0); // dl/db3 = d/b3 (categorical_crossentropy) = dl_dz3 dz3/b3
            Tensor dl_db2 = sum(dl_dz2, 0); // dl/db2 = d/b2 (categorical_crossentropy) = dl_dz2 dz2/b2
            Tensor dl_db1 = sum(dl_dz1, 0); // dl/db1 = d/b1 (categorical_crossentropy) = dl_dz1 dz1/b1

            #if GRADIENT_CLIPPING_ENABLED
                dl_dw3 = clip_by_value(dl_dw3, -GRADIENT_CLIP_THRESHOLD, GRADIENT_CLIP_THRESHOLD);
                dl_dw2 = clip_by_value(dl_dw2, -GRADIENT_CLIP_THRESHOLD, GRADIENT_CLIP_THRESHOLD);
                dl_dw1 = clip_by_value(dl_dw1, -GRADIENT_CLIP_THRESHOLD, GRADIENT_CLIP_THRESHOLD);

                dl_db3 = clip_by_value(dl_db3, -GRADIENT_CLIP_THRESHOLD, GRADIENT_CLIP_THRESHOLD);
                dl_db2 = clip_by_value(dl_db2, -GRADIENT_CLIP_THRESHOLD, GRADIENT_CLIP_THRESHOLD);
                dl_db1 = clip_by_value(dl_db1, -GRADIENT_CLIP_THRESHOLD, GRADIENT_CLIP_THRESHOLD);
            #endif

            // TODO: Don't I have to add regularizer for the biases as well like tf.keras.layers.Dense does?
            // TODO: What is kernel_constraint and bias_constraint?
            #if MOMENTUM_ENABLED
                w_m[2] = MOMENTUM * w_m[2] - LEARNING_RATE * dl_dw3;
                w_m[1] = MOMENTUM * w_m[1] - LEARNING_RATE * dl_dw2;
                w_m[0] = MOMENTUM * w_m[0] - LEARNING_RATE * dl_dw1;

                b_m[2] = MOMENTUM * b_m[2] - LEARNING_RATE * dl_db3;
                b_m[1] = MOMENTUM * b_m[1] - LEARNING_RATE * dl_db2;
                b_m[0] = MOMENTUM * b_m[0] - LEARNING_RATE * dl_db1;

                #if 1 // Standard
                    for (short i = LAYERS.size() - 2; 0 <= i; --i) {
                        w[i] += w_m[i];
                        b[i] += b_m[i];
                    }
                #endif

                // TODO: Handle nestrov for momentum.
                #if 0 // Nesterov
                #endif
            #else
                w[2] -= LEARNING_RATE * dl_dw3;
                w[1] -= LEARNING_RATE * dl_dw2;
                w[0] -= LEARNING_RATE * dl_dw1;

                b[2] -= LEARNING_RATE * dl_db3;
                b[1] -= LEARNING_RATE * dl_db2;
                b[0] -= LEARNING_RATE * dl_db1;
            #endif*/
        }

        #define LOG_EPOCH(i, EPOCHS) std::cout << "Epoch " << (i) << "/" << (EPOCHS)

        #if L1_REGULARIZATION_ENABLED || L2_REGULARIZATION_ENABLED || L1L2_REGULARIZATION_ENABLED
            LOG_EPOCH(i, EPOCHS);
            log_metrics("training", y_batch, a.back(), &w);
        #else
            LOG_EPOCH(i, EPOCHS);
            log_metrics("training", y_batch, a.back());
        #endif

        a = forward_propagation(val_test.x_first, w, b);

        log_metrics("val", val_test.y_first, a.back());
        std::cout << std::endl;

        // TODO: It seems like early stopping is legit regularization so it might be wise to write this function in regularizer files.
        #if EARLY_STOPPING_ENABLED
            static unsigned short epochs_without_improvement = 0;
            static float best_val_loss = std::numeric_limits<float>::max();

            if (LOSS(val_test.y_first, a.back()) < best_val_loss) {
                best_val_loss = LOSS(val_test.y_first, a.back());
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
            }

            if (epochs_without_improvement >= PATIENCE) {
                std::cout << std::endl << "Early stopping at epoch " << i + 1 << " as validation loss did not improve for " << PATIENCE << " epochs." << std::endl;
                break;
            }
        #endif
    }

    auto a = forward_propagation(val_test.x_second, w, b);

    std::cout << std::endl;
    log_metrics("test", val_test.y_second, a.back());
    std::cout << std::endl << std::endl;

    std::cout << a.back() << std::endl << std::endl << val_test.y_second;
}