#include "datas.h"
#include "lyrs.h"

// 1. load_imdb() has to return vectorized tensor like tf.keras.datasets.imdb.load_data()
//     - Continue use imdb dataset downloaded from Kaggle websites as instead of the datasets used for
//     tf.keras.datasets.imdb.load_data() as it is conversome and the datasets has nothing to do with the order it's
//     just random as in the order of the dataset from tf.keras.datasets.imdb.load_data() and the dataset downloaded
//     from the link on the tf.keras.datasets.imdb.load_data() don't much.
// 2. Implement Vanilla RNNs which are simplest form of RNNs. Has only single hidden layer.
//     - Implement forward prop for many-to-one and many-to-many as these are more common. One-to-one and one-to-many
//     are
// less common. One-to-one might be just regular normal neural netwok when you think about it...
//     - Implement backprop for all the correspondence for above.
// 3. Implement LSTM
// 4. Implement GRU
// 5. Implement Bidirectional RNNs
// 6. Implement Deep RNNs which have multiple layers of RNNs stacked on top of each other, and can be built with any of
// the basic RNN units (vanilla, LSTM, GRU)

int main()
{
    auto data = load_imdb();

    std::cout << "done" << std::endl;

    return 0;
}