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
#include <math.h>

constexpr std::array<unsigned short, 3> LAYERS = { 1, 64, 1 };

constexpr unsigned short EPOCHS = 1;
float LEARNING_RATE             = 0.1f;
unsigned int SEQUENCE_LENGTH    = 12; // Use the previous 12 months to predict the next month

// Tensor make_sequences_labels(Tensor& tensor const Tensor& tensor) {
//     int idx = 0;
//     for (int i = 0; i < (train_test.x_first._size - SEQUENCE_LENGTH) * SEQUENCE_LENGTH; ++i) {
//         if (i % SEQUENCE_LENGTH == 0 && i != 0)
//             idx -= SEQUENCE_LENGTH - 1;
//         tensor[i] = train_test.x_first[idx];
//         // x_train[i] = i <= SEQUENCE_LENGTH ? train_test.x_first[idx] : train_test.x_first[idx - SEQUENCE_LENGTH - 1];
//         ++idx;
//     }
// }

// TODO: I could try a dataset like Air Quality.
int main() {
    // Load the Air Passengers dataset
    Tensor air_passengers = load_air_passengers();

    // Normalize the dataset (scaling to [0, 1] range)
    air_passengers = min_max_scaler(air_passengers);

    // Split the data into training and testing sets
    TrainTest train_test = train_test_split(air_passengers, 0.33, 42); // TODO: Don't forget to shuffle in train_test_split!

    // Create sequences and corresponding labels for training
    // TODO: Maybe I don't need to make 3d tensor as x eventually I'm extracting x as in 2d tensor which just rewinding what I'm doing in below for loops. I mean it llooks like it's all about matrix multipication which is 2d?
    Tensor x_train = Tensor({ 0.0f }, { train_test.x_first._size - SEQUENCE_LENGTH, SEQUENCE_LENGTH, 1 });
    Tensor y_train = Tensor({ 0.0f }, { train_test.x_first._size - SEQUENCE_LENGTH, 1 });
    Tensor x_test  = Tensor({ 0.0f }, { train_test.x_second._size - SEQUENCE_LENGTH, SEQUENCE_LENGTH, 1 });
    Tensor y_test  = Tensor({ 0.0f }, { train_test.x_second._size - SEQUENCE_LENGTH, 1 });

    // TODO: Prepare validation dataset, and review what is the best relation with train dataset

    int idx = 0;
    for (int i = 0; i < (train_test.x_first._size - SEQUENCE_LENGTH) * SEQUENCE_LENGTH; ++i) {
        if (i % SEQUENCE_LENGTH == 0 && i != 0)
            idx -= SEQUENCE_LENGTH - 1;
        x_train[i] = train_test.x_first[idx];
        // x_train[i] = i <= SEQUENCE_LENGTH ? train_test.x_first[idx] : train_test.x_first[idx - SEQUENCE_LENGTH - 1];
        ++idx;
    }

    for (int i = 0; i < train_test.x_first._size - SEQUENCE_LENGTH; ++i) {
        y_train[i] = train_test.x_first[i + SEQUENCE_LENGTH];
    }

    idx = 0;
    for (int i = 0; i < (train_test.x_second._size - SEQUENCE_LENGTH) * SEQUENCE_LENGTH; ++i) {
        if (i % SEQUENCE_LENGTH == 0 && i != 0)
            idx -= SEQUENCE_LENGTH - 1;
        x_test[i] = train_test.x_second[idx];
        // x_train[i] = i <= SEQUENCE_LENGTH ? train_test.x_first[idx] : train_test.x_first[idx - SEQUENCE_LENGTH - 1];
        ++idx;
    }

    for (int i = 0; i < train_test.x_second._size - SEQUENCE_LENGTH; ++i) {
        y_test[i] = train_test.x_second[i + SEQUENCE_LENGTH];
    }

    // Initialize weights and biases
    Tensor wxh = normal_distribution({ LAYERS[1], LAYERS[0] });
    Tensor whh = normal_distribution({ LAYERS[1], LAYERS[1] });
    Tensor why = normal_distribution({ LAYERS[2], LAYERS[1] });

    Tensor bh = zeros({ LAYERS[1], 1 });
    Tensor by = zeros({ 1, 1 });

    for (int i = 1; i <= EPOCHS; ++i) {
        Tensor hideen_state = zeros({ LAYERS[1], 1 });
        unsigned int idx = 0;

        for (int j = 0; j < x_train._shape[0]; ++j) {
            Tensor x = Tensor({ 0.0f }, { SEQUENCE_LENGTH, 1 });

            for (int k = 0; k < SEQUENCE_LENGTH; ++k) {
                x[k] = x_train[idx];
                ++idx;
            }

            float target = y_train[j];

            // Forward propagation
            // TODO: tanh should be placed in activations file?
            Tensor h      = tanh(matmul(wxh, x) + matmul(whh, hprev) + bh);
            Tensor y_pred = matmul(why, h) + by;
            // h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, hprev) + bh)
            // y_pred = np.dot(Why, h) + by

            std::cout << x << std::endl;
            std::cout << target << std::endl;
        }
    }
    
    std::cout << x_train << std::endl;

    //   x = tf.constant([-float("inf"), -5, -0.5, 1, 1.2, 2, 3, float("inf")])
    Tensor fd = Tensor({ -INFINITY, -5, -0.5, 1, 1.2, 2, 3, INFINITY }, { 1, 8 });
    std::cout << tanh(fd) << std::endl;

    // TODO: use auto for for loop?
}