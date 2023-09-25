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

// TODO: I could try a dataset like Air Quality.
int main() {
    // Load the Air Passengers dataset
    Tensor air_passengers = load_air_passengers();

    // Normalize the dataset (scaling to [0, 1] range)
    air_passengers = min_max_scaler(air_passengers);

    // Split the data into training and testing sets
    TrainTest train_temp = train_test_split(air_passengers, 0.33, 42); // TODO: Don't forget to shuffle in train_test_split!

    // Create sequences and corresponding labels for training
    // short sequence_length = 12 // Use the previous 12 months to predict the next month

    // for i in range(len(train_data) - sequence_length):
    //     X_train.append(train_data[i:i + sequence_length])
    //     y_train.append(train_data[i + sequence_length])

    // X_train, y_train = np.array(X_train), np.array(y_train)

    std::cout << train_temp.x_first << std::endl;
    std::cout << train_temp.x_second << std::endl;
}