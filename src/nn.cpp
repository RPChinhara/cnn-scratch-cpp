#include "nn.h"
#include "activations.h"
#include "arrays.h"
#include "derivatives.h"
#include "linalg.h"
#include "mathematics.h"
#include "random.h"
#include "regularizations.h"

#include <random>

NN::NN(const std::vector<unsigned int>& layers, float learning_rate) {
    this->layers = layers;
    this->learning_rate = learning_rate;
}

void NN::train(const Tensor& train_x, const Tensor& train_y, const Tensor& val_x, const Tensor& val_y) {
    // Init parameters
    w_b = init_parameters();
    #if MOMENTUM_ENABLED
        w_b_m = init_parameters();
    #endif

    for (unsigned short i = 1; i <= epochs; ++i) {
        // Set learning rate scheduler
        #if LEARNING_RATE_SCHEDULER_ENABLED
            if (i > 10 && i < 20)      learning_rate = 0.009f;
            else if (i > 20 && i < 30) learning_rate = 0.005f;
            else                       learning_rate = 0.001f;
        #endif

        // Shuffle the dataset
        std::random_device rd;
        auto rd_num = rd();
        Tensor x_shuffled = shuffle(train_x, rd_num);
        Tensor y_shuffled = shuffle(train_y, rd_num);

        Tensor y_batch;
        TensorArray  a;

        // SGD (Mini-batch gradient descent)
        for (unsigned int j = 0; j < train_x._shape.front(); j += batch_size) {
            // Slice the dataset previously shuffled
            Tensor x_batch = slice(x_shuffled, j, batch_size);
            y_batch        = slice(y_shuffled, j, batch_size);

            // regularizer::dropout(0.1f, x);
            a = forward_propagation(x_batch, w_b.first, w_b.second);
            
            // Backpropagation
            std::vector<Tensor> dl_dz, dl_dw, dl_db;

            // Calculate dl/dz
            for (unsigned char k = layers.size() - 1; k > 0; --k) {
                // dl/dz3 = dl/dy dy/dz3
                // dl/dz2 = dl/dz3 dz3/da2 da2/z2
                // dl/dz1 = dl/dz2 dz2/da1 da1/z1

                if (k == layers.size() - 1)
                    dl_dz.push_back(categorical_crossentropy_prime(y_batch, a.back()));
                else
                    dl_dz.push_back(matmul(dl_dz[(layers.size() - 2) - k], w_b.first[k].T()) * relu_prime(a[k - 1]));
            }

            // Calculate dl/dw
            for (unsigned char k = layers.size() - 1; k > 0; --k) {
                // dl/dw3 = dl/dz3 dz3/dw3 (+ dl1/w3 or + dl2/w3 or + dl1/w3 + dl2/w3)
                // dl/dw2 = dl/dz2 dz2/dw2 (+ dl1/w2 or + dl2/w2 or + dl1/w2 + dl2/w2)
                // dl/dw1 = dl/dz1 dz1/dw1 (+ dl1/w1 or + dl2/w1 or + dl1/w1 + dl2/w1)

                if (k == 1) {
                    #if L1_REGULARIZATION_ENABLED && !L2_REGULARIZATION_ENABLED && !L1L2_REGULARIZATION_ENABLED
                        dl_dw.push_back(matmul(x_batch.T(), dl_dz[(layers.size() - 1) - k]) + l1_prime(l1_lambda, w_b.first[0]));
                    #elif L2_REGULARIZATION_ENABLED && !L1_REGULARIZATION_ENABLED && !L1L2_REGULARIZATION_ENABLED
                        dl_dw.push_back(matmul(x_batch.T(), dl_dz[(layers.size() - 1) - k]) + l2_prime(l2_lambda, w_b.first[0]));
                    #elif L1L2_REGULARIZATION_ENABLED && !L1_REGULARIZATION_ENABLED && !L2_REGULARIZATION_ENABLED
                        dl_dw.push_back(matmul(x_batch.T(), dl_dz[(layers.size() - 1) - k]) + l1_prime(l1_lambda, w_b.first[0]) + l2_prime(l2_lambda, w_b.first[0]));
                    #else
                        dl_dw.push_back(matmul(x_batch.T(), dl_dz[(layers.size() - 1) - k]));
                    #endif
                } else {
                    #if L1_REGULARIZATION_ENABLED && !L2_REGULARIZATION_ENABLED && !L1L2_REGULARIZATION_ENABLED
                        dl_dw.push_back(matmul(a[k - 2].T(), dl_dz[(layers.size() - 1) - k]) + l1_prime(l1_lambda, w_b.first[k - 1]));
                    #elif L2_REGULARIZATION_ENABLED && !L1_REGULARIZATION_ENABLED && !L1L2_REGULARIZATION_ENABLED
                        dl_dw.push_back(matmul(a[k - 2].T(), dl_dz[(layers.size() - 1) - k]) + l2_prime(l2_lambda, w_b.first[k - 1]));
                    #elif L1L2_REGULARIZATION_ENABLED && !L1_REGULARIZATION_ENABLED && !L2_REGULARIZATION_ENABLED
                        dl_dw.push_back(matmul(a[k - 2].T(), dl_dz[(layers.size() - 1) - k]) + l1_prime(l1_lambda, w_b.first[k - 1]) + l2_prime(l2_lambda, w_b.first[k - 1]));
                    #else
                        dl_dw.push_back(matmul(a[k - 2].T(), dl_dz[(layers.size() - 1) - k]));
                    #endif
                }
            }

            // Calculate dl/db
            for (unsigned char k = 0; k < layers.size() - 1; ++k) {
                // dl/db3 = dl/dz3 dz3/b3
                // dl/db2 = dl/dz2 dz2/b2
                // dl/db1 = dl/dz1 dz1/b1
                
                dl_db.push_back(sum(dl_dz[k], 0));
            }

            #if GRADIENT_CLIPPING_ENABLED
                for (unsigned char k = 0; k < layers.size() - 1; ++k) {
                    dl_dw[k] = clip_by_value(dl_dw[k], -gradient_clip_threshold, gradient_clip_threshold);
                    dl_db[k] = clip_by_value(dl_db[k], -gradient_clip_threshold, gradient_clip_threshold);
                }
            #endif

            // Updating the parameters
            #if !MOMENTUM_ENABLED
                for (char k = layers.size() - 2; k >= 0; --k) {
                    w_b.first[k]  -= learning_rate * dl_dw[(layers.size() - 2) - k];
                    w_b.second[k] -= learning_rate * dl_db[(layers.size() - 2) - k];
                }
            #else
                for (char k = layers.size() - 2; k >= 0; --k) {
                    w_b_m.first[k]  = momentum * w_b_m.first[k] - learning_rate * dl_dw[(layers.size() - 2) - k];
                    w_b_m.second[k] = momentum * w_b_m.second[k] - learning_rate * dl_db[(layers.size() - 2) - k];
                }

                #if 1 // Standard
                    for (char k = layers.size() - 2; k >= 0; --k) {
                        w_b.first[k]  += w_b_m.first[k];
                        w_b.second[k] += w_b_m.second[k];
                    }
                #else // Nesterov
                #endif
            #endif
        }

        // Logging the metrics
        #define LOG_EPOCH(i, EPOCH) std::cout << "Epoch " << (i) << "/" << (EPOCH)

        #if L1_REGULARIZATION_ENABLED || L2_REGULARIZATION_ENABLED || L1L2_REGULARIZATION_ENABLED
            LOG_EPOCH(i, epochs);
            log_metrics("training", y_batch, a.back(), &w_b.first);
        #else
            LOG_EPOCH(i, epochs);
            log_metrics("training", y_batch, a.back());
        #endif

        a = forward_propagation(val_x, w_b.first, w_b.second);

        log_metrics("val", val_y, a.back());
        std::cout << std::endl;

        // Early stopping
        #if EARLY_STOPPING_ENABLED
            static unsigned char epochs_without_improvement = 0;
            static float best_val_loss = std::numeric_limits<float>::max();

            if (LOSS(val_y, a.back()) < best_val_loss) {
                best_val_loss = LOSS(val_y, a.back());
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
            }

            if (epochs_without_improvement >= patience) {
                std::cout << std::endl << "Early stopping at epoch " << i + 1 << " as validation loss did not improve for " << static_cast<unsigned short>(patience) << " epochs." << std::endl;
                break;
            }
        #endif
    }
}

void NN::predict(const Tensor& test_x, const Tensor& test_y) {
    auto a = forward_propagation(test_x, w_b.first, w_b.second);

    // Logging the metrics
    std::cout << std::endl;
    log_metrics("test", test_y, a.back());
    std::cout << std::endl << std::endl;

    // Comparing the y_train and y_test
    std::cout << a.back() << std::endl << std::endl << test_y << std::endl;
}

TensorArray NN::forward_propagation(const Tensor& input, const TensorArray& w, const TensorArray& b) {
    TensorArray z;
    TensorArray a;

    for (unsigned char i = 0; i < layers.size() - 1; ++i) {
        if (i == 0) {
            z.push_back((matmul(input, w[i]) + b[i]));
            a.push_back((relu(z[i])));
        } else {
            z.push_back((matmul(a[i - 1], w[i]) + b[i]));
            if (i == 1)
                a.push_back((softmax(z[i])));
        }
    }

    return a;
}

std::pair<TensorArray, TensorArray> NN::init_parameters() {
    TensorArray w;
    TensorArray b;

    for (unsigned int i = 0; i < layers.size() - 1; ++i) {
        w.push_back(normal_distribution({ layers[i], layers[i + 1] }, 0.0f, 2.0f));
        b.push_back(zeros({ 1, layers[i + 1] }));
    }

    return std::make_pair(w, b);
}

void NN::log_metrics(const std::string& data, const Tensor& y_true, const Tensor& y_pred, const TensorArray *w) {
    if (data != "test") {
        if (!w) {
            std::cout << " - " << data << " loss: " << LOSS(y_true, y_pred) << " - " << data << " accuracy: " << ACCURACY(y_true, y_pred);
        } else {
            #if L1_REGULARIZATION_ENABLED && !L2_REGULARIZATION_ENABLED && !L1L2_REGULARIZATION_ENABLED
                float l1 = 0.0f;

                for (unsigned char i = 0; i < layers.size() - 1; ++i)
                    l1 += l1(l1_lambda, (*w)[i]);

                std::cout << " - " << data << " loss: " << LOSS(y_true, y_pred) + l1 << " - " << data << " accuracy: " << ACCURACY(y_true, y_pred);
            #elif L2_REGULARIZATION_ENABLED && !L1_REGULARIZATION_ENABLED && !L1L2_REGULARIZATION_ENABLED
                float l2 = 0.0f;

                for (unsigned char i = 0; i < layers.size() - 1; ++i)
                    l2 += l2(l2_lambda, (*w)[i]);

                std::cout << " - " << data << " loss: " << LOSS(y_true, y_pred) + l2 << " - " << data << " accuracy: " << ACCURACY(y_true, y_pred);
            #elif L1L2_REGULARIZATION_ENABLED && !L1_REGULARIZATION_ENABLED && !L2_REGULARIZATION_ENABLED
                float l1l2 = 0.0f;
                
                for (unsigned char i = 0; i < layers.size() - 1; ++i) 
                    l1l2 += l1(l1_lambda, (*w)[i]) + l2(l2_lambda, (*w)[i]);
                
                std::cout << " - " << data << " loss: " << LOSS(y_true, y_pred) + l1l2 << " - " << data << " accuracy: " << ACCURACY(y_true, y_pred);
            #else
                std::cerr << std::endl << __FILE__ << "(" << __LINE__ << ")" << ": error: choose only one regularization" << std::endl;
                exit(1);
            #endif
        }
    } else {
        std::cout << data << " loss: " << LOSS(y_true, y_pred) << " - " << data << " accuracy: " << ACCURACY(y_true, y_pred);
    }
}