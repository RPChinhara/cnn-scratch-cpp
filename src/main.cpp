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
#include <windows.h>
// #pragma comment (lib, "User32.lib")

// Hyperparameters
#define EARLY_STOPPING_ENABLED          1
#define GRADIENT_CLIPPING_ENABLED       1
#define LEARNING_RATE_SCHEDULER_ENABLED 1
#define L1_REGULARIZATION_ENABLED       0
#define L2_REGULARIZATION_ENABLED       0
#define L1L2_REGULARIZATION_ENABLED     1
#define MOMENTUM_ENABLED                1

constexpr std::array<unsigned short, 3> LAYERS = { 4, 128, 3 };

constexpr auto ACCURACY = &categorical_accuracy;
constexpr auto LOSS     = &categorical_crossentropy;

constexpr unsigned short BATCH_SIZE = 8;
constexpr unsigned short EPOCHS     = 100;
float LEARNING_RATE                 = 0.01f;

[[maybe_unused]] constexpr float GRADIENT_CLIP_THRESHOLD = 8.0f;
[[maybe_unused]] constexpr float MOMENTUM                = 0.1f;
[[maybe_unused]] constexpr unsigned char PATIENCE        = 12;

[[maybe_unused]] constexpr float L1_LAMBDA = 0.01f;
[[maybe_unused]] constexpr float L2_LAMBDA = 0.01f;

[[maybe_unused]] constexpr float BETA_1  = 0.9f;
[[maybe_unused]] constexpr float BETA_2  = 0.999f;
[[maybe_unused]] constexpr float EPSILON = 1e-8f;
[[maybe_unused]] constexpr float M_T     = 0;
[[maybe_unused]] constexpr float V_T     = 0;
[[maybe_unused]] constexpr float T       = 0;

using TensorArray = std::array<Tensor, LAYERS.size() - 1>;

TensorArray forward_propagation(const Tensor& input, const TensorArray& w, const TensorArray& b) {
    TensorArray z;
    TensorArray a;

    for (unsigned char i = 0; i < LAYERS.size() - 1; ++i) {
        if (i == 0) {
            z[i] = (matmul(input, w[i]) + b[i]);
            a[i] = (relu(z[i]));
        } else {
            z[i] = (matmul(a[i - 1], w[i]) + b[i]);
            if (i == 1)
                a[i] = (softmax(z[i]));
        }
    }

    return a;
}

// TODO: maybe rename evetything to use camel case so that it'd much with Win API?
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
                float l1 = 0.0f;

                for (unsigned char i = 0; i < LAYERS.size() - 1; ++i)
                    l1 += l1(L1_LAMBDA, (*w)[i]);

                std::cout << " - " << data << " loss: " << LOSS(y_true, y_pred) + l1 << " - " << data << " accuracy: " << ACCURACY(y_true, y_pred);
            #elif L2_REGULARIZATION_ENABLED && !L1_REGULARIZATION_ENABLED && !L1L2_REGULARIZATION_ENABLED
                float l2 = 0.0f;

                for (unsigned char i = 0; i < LAYERS.size() - 1; ++i)
                    l2 += l2(L2_LAMBDA, (*w)[i]);

                std::cout << " - " << data << " loss: " << LOSS(y_true, y_pred) + l2 << " - " << data << " accuracy: " << ACCURACY(y_true, y_pred);
            #elif L1L2_REGULARIZATION_ENABLED && !L1_REGULARIZATION_ENABLED && !L2_REGULARIZATION_ENABLED
                float l1l2 = 0.0f;
                
                for (unsigned char i = 0; i < LAYERS.size() - 1; ++i) 
                    l1l2 += l1(L1_LAMBDA, (*w)[i]) + l2(L2_LAMBDA, (*w)[i]);
                
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

LRESULT CALLBACK WindowProcedure(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch(uMsg) {
        case WM_CLOSE:
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    // Create a console for logging otherwise I can't when WinMain() is used as the entry point because it doesn't use the standard console for input and output
    AllocConsole();
    freopen_s((FILE**)stdout, "CONOUT$", "w", stdout);

    // Load the Iris dataset
    Iris iris = load_iris();
    Tensor x = iris.features;
    Tensor y = iris.target;

    // Preprocess the dataset
    y = one_hot(y, 3);
    TrainTest2 train_temp = train_test_split(x, y, 0.2, 42);
    TrainTest2 val_test   = train_test_split(train_temp.x_second, train_temp.y_second, 0.5, 42);
    train_temp.x_first = min_max_scaler(train_temp.x_first);
    val_test.x_first   = min_max_scaler(val_test.x_first);
    val_test.x_second  = min_max_scaler(val_test.x_second);

    // Init parameters
    auto w_b = init_parameters();
    #if MOMENTUM_ENABLED
        auto w_b_m = init_parameters();
    #endif

    // Training
    for (unsigned short i = 1; i <= EPOCHS; ++i) {
        // TODO: Try batch normalization again (I saw it was used in SOTA model in a paper so I might need to work on this).
        // TODO: Use cross-validation technique?

        // Set learning rate scheduler
        #if LEARNING_RATE_SCHEDULER_ENABLED
            if (i > 10 && i < 20)      LEARNING_RATE = 0.009f;
            else if (i > 20 && i < 30) LEARNING_RATE = 0.005f;
            else                       LEARNING_RATE = 0.001f;
        #endif

        // Shuffle the dataset
        std::random_device rd;
        auto rd_num = rd();
        Tensor x_shuffled = shuffle(train_temp.x_first, rd_num);
        Tensor y_shuffled = shuffle(train_temp.y_first, rd_num);

        Tensor y_batch;
        TensorArray  a;

        // SGD (Mini-batch gradient descent)
        for (unsigned int j = 0; j < train_temp.x_first._shape.front(); j += BATCH_SIZE) {
            // TODO: For loop used for mimi batch gradient process multiple examples in parallel utilizing GPUs. That's the main reason facilitating mini-batch training (use std::thread).
            
            // Slice the dataset previously shuffled
            Tensor x_batch = slice(x_shuffled, j, j + BATCH_SIZE);
            y_batch        = slice(y_shuffled, j, j + BATCH_SIZE);

            // regularizer::dropout(0.1f, x);
            a = forward_propagation(x_batch, w_b.first, w_b.second);
            
            // Backpropagation
            std::vector<Tensor> dl_dz, dl_dw, dl_db;

            // TODO: Implement Adam and AdamW.

            // Calculate dl/dz
            for (unsigned char k = LAYERS.size() - 1; 0 < k; --k) {
                // TODO: Don't I really have to multiply by relu_prime for dy/dz3?
                // dl/dz3 = dl/dy dy/dz3
                // dl/dz2 = dl/dz3 dz3/da2 da2/z2
                // dl/dz1 = dl/dz2 dz2/da1 da1/z1

                if (k == LAYERS.size() - 1)
                    dl_dz.push_back(categorical_crossentropy_prime(y_batch, a.back()));
                else
                    dl_dz.push_back(matmul(dl_dz[(LAYERS.size() - 2) - k], w_b.first[k].T()) * relu_prime(a[k - 1]));

                // TODO: I could use above '(LAYERS.size() - 2) - i' so that I don't have to use idx, and this applies to other functions use idx. 
            }

            // Calculate dl/dw
            for (unsigned char k = LAYERS.size() - 1; 0 < k; --k) {
                // dl/dw3 = dl/dz3 dz3/dw3 (+ dl1/w3 or + dl2/w3 or + dl1/w3 + dl2/w3)
                // dl/dw2 = dl/dz2 dz2/dw2 (+ dl1/w2 or + dl2/w2 or + dl1/w2 + dl2/w2)
                // dl/dw1 = dl/dz1 dz1/dw1 (+ dl1/w1 or + dl2/w1 or + dl1/w1 + dl2/w1)

                if (k == 1) {
                    #if L1_REGULARIZATION_ENABLED && !L2_REGULARIZATION_ENABLED && !L1L2_REGULARIZATION_ENABLED
                        dl_dw.push_back(matmul(x_batch.T(), dl_dz[(LAYERS.size() - 1) - k]) + l1_prime(L1_LAMBDA, w_b.first[0]));
                    #elif L2_REGULARIZATION_ENABLED && !L1_REGULARIZATION_ENABLED && !L1L2_REGULARIZATION_ENABLED
                        dl_dw.push_back(matmul(x_batch.T(), dl_dz[(LAYERS.size() - 1) - k]) + l2_prime(L2_LAMBDA, w_b.first[0]));
                    #elif L1L2_REGULARIZATION_ENABLED && !L1_REGULARIZATION_ENABLED && !L2_REGULARIZATION_ENABLED
                        dl_dw.push_back(matmul(x_batch.T(), dl_dz[(LAYERS.size() - 1) - k]) + l1_prime(L1_LAMBDA, w_b.first[0]) + l2_prime(L2_LAMBDA, w_b.first[0]));
                    #else
                        dl_dw.push_back(matmul(x_batch.T(), dl_dz[(LAYERS.size() - 1) - k]));
                    #endif
                } else {
                    #if L1_REGULARIZATION_ENABLED && !L2_REGULARIZATION_ENABLED && !L1L2_REGULARIZATION_ENABLED
                        dl_dw.push_back(matmul(a[k - 2].T(), dl_dz[(LAYERS.size() - 1) - k]) + l1_prime(L1_LAMBDA, w_b.first[k - 1]));
                    #elif L2_REGULARIZATION_ENABLED && !L1_REGULARIZATION_ENABLED && !L1L2_REGULARIZATION_ENABLED
                        dl_dw.push_back(matmul(a[k - 2].T(), dl_dz[(LAYERS.size() - 1) - k]) + l2_prime(L2_LAMBDA, w_b.first[k - 1]));
                    #elif L1L2_REGULARIZATION_ENABLED && !L1_REGULARIZATION_ENABLED && !L2_REGULARIZATION_ENABLED
                        dl_dw.push_back(matmul(a[k - 2].T(), dl_dz[(LAYERS.size() - 1) - k]) + l1_prime(L1_LAMBDA, w_b.first[k - 1]) + l2_prime(L2_LAMBDA, w_b.first[k - 1]));
                    #else
                        dl_dw.push_back(matmul(a[k - 2].T(), dl_dz[(LAYERS.size() - 1) - k]));
                    #endif
                }
            }

            // Calculate dl/db
            for (unsigned char k = 0; k < LAYERS.size() - 1; ++k) {
                // dl/db3 = dl/dz3 dz3/b3
                // dl/db2 = dl/dz2 dz2/b2
                // dl/db1 = dl/dz1 dz1/b1
                
                dl_db.push_back(sum(dl_dz[k], 0));
            }

            #if GRADIENT_CLIPPING_ENABLED
                for (unsigned char k = 0; k < LAYERS.size() - 1; ++k) {
                    dl_dw[k] = clip_by_value(dl_dw[k], -GRADIENT_CLIP_THRESHOLD, GRADIENT_CLIP_THRESHOLD);
                    dl_db[k] = clip_by_value(dl_db[k], -GRADIENT_CLIP_THRESHOLD, GRADIENT_CLIP_THRESHOLD);
                }
            #endif

            // TODO: Don't I have to add regularizer for the biases as well like tf.keras.layers.Dense does?
            // TODO: What is kernel_constraint and bias_constraint?
            // TODO: I could create a file called optimizer, and put below codes as SDG same for upcoming Adam and AdamW.

            // Updating the parameters
            #if !MOMENTUM_ENABLED
                for (char k = LAYERS.size() - 2; 0 <= k; --k) {
                    w_b.first[k]  -= LEARNING_RATE * dl_dw[(LAYERS.size() - 2) - k];
                    w_b.second[k] -= LEARNING_RATE * dl_db[(LAYERS.size() - 2) - k];
                }
            #else
                for (char k = LAYERS.size() - 2; 0 <= k; --k) {
                    w_b_m.first[k]  = MOMENTUM * w_b_m.first[k] - LEARNING_RATE * dl_dw[(LAYERS.size() - 2) - k];
                    w_b_m.second[k] = MOMENTUM * w_b_m.second[k] - LEARNING_RATE * dl_db[(LAYERS.size() - 2) - k];
                }

                #if 1 // Standard
                    for (char k = LAYERS.size() - 2; 0 <= k; --k) {
                        w_b.first[k]  += w_b_m.first[k];
                        w_b.second[k] += w_b_m.second[k];
                    }
                #else // Nesterov
                    // TODO: Handle nestrov for momentum.
                #endif
            #endif
        }

        // TODO: Make progress bar.

        // Logging the metrics
        #define LOG_EPOCH(i, EPOCHS) std::cout << "Epoch " << (i) << "/" << (EPOCHS)

        #if L1_REGULARIZATION_ENABLED || L2_REGULARIZATION_ENABLED || L1L2_REGULARIZATION_ENABLED
            LOG_EPOCH(i, EPOCHS);
            log_metrics("training", y_batch, a.back(), &w_b.first);
        #else
            LOG_EPOCH(i, EPOCHS);
            log_metrics("training", y_batch, a.back());
        #endif

        a = forward_propagation(val_test.x_first, w_b.first, w_b.second);

        log_metrics("val", val_test.y_first, a.back());
        std::cout << std::endl;

        // TODO: It seems like early stopping is legit regularization so it might be wise to write this function in regularizer files.

        // Early stopping
        #if EARLY_STOPPING_ENABLED
            static unsigned char epochs_without_improvement = 0;
            static float best_val_loss = std::numeric_limits<float>::max();

            if (LOSS(val_test.y_first, a.back()) < best_val_loss) {
                best_val_loss = LOSS(val_test.y_first, a.back());
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
            }

            if (epochs_without_improvement >= PATIENCE) {
                std::cout << std::endl << "Early stopping at epoch " << i + 1 << " as validation loss did not improve for " << static_cast<unsigned short>(PATIENCE) << " epochs." << std::endl;
                break;
            }
        #endif
    }

    // Prediction
    auto a = forward_propagation(val_test.x_second, w_b.first, w_b.second);

    // Logging the metrics
    std::cout << std::endl;
    log_metrics("test", val_test.y_second, a.back());
    std::cout << std::endl << std::endl;

    // Comparing the y_train and y_test
    std::cout << a.back() << std::endl << std::endl << val_test.y_second << std::endl;

    // Logging Biological conditions
    // while (true)   
    //     std::cout << "Life: 100 - Hunger:100 - Height: 175cm - Weight: 65kg - Thirstiness: 100" << std::endl;
    
    // Q-learning
    unsigned int n_observations = 16;
    unsigned int n_actions      = 4;
    
    // Initialize the Q-table to 0
    Tensor q_table = zeros({ n_observations, n_actions });
    std::cout << q_table << std::endl;

    // number of episode we will run
    int n_episodes = 10000;

    // maximum of iteration per episode
    int max_iter_episode = 100;

    // initialize the exploration probability to 1
    int exploration_proba = 1;

    // exploartion decreasing decay for exponential decreasing
    float exploration_decreasing_decay = 0.001f;

    // minimum of exploration proba
    float min_exploration_proba = 0.01f;

    // discounted factor
    float gamma = 0.99f;

    float lr = 0.1f;

    // Making the window
    const char CLASS_NAME[] = "Sample Window Class";

    WNDCLASS wc = {};
    wc.lpfnWndProc = WindowProcedure;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(
        0,
        CLASS_NAME,
        "Sample Window",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
        NULL,
        NULL,
        hInstance,
        NULL
    );

    if (hwnd == NULL) {
        return 0;
    }

    ShowWindow(hwnd, nCmdShow);

    MSG msg = {};
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}