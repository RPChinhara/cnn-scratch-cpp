#include "datas.h"
#include "lyrs.h"

// 1. load_imdb() has to return a vectorized tensor like tf.keras.datasets.imdb.load_data().
//     - Continue using the IMDb dataset downloaded from Kaggle instead of the datasets used for
//     tf.keras.datasets.imdb.load_data() as it is cumbersome, and the datasets have nothing to do with the order.
//     The order of the dataset from tf.keras.datasets.imdb.load_data() and the dataset downloaded
//     from the link on the tf.keras.datasets.imdb.load_data() don't match.
// 2. Implement Vanilla RNNs, which are the simplest form of RNNs. They have only a single hidden layer.
//     - Implement forward propagation for many-to-one and many-to-many, as these are more common. One-to-one and
//     one-to-many are less common. One-to-one might just be a regular neural network when you think about it...
//     - Implement backpropagation for all the cases mentioned above.
// 3. Implement LSTM.
// 4. Implement GRU.
// 5. Implement Bidirectional RNNs.
// 6. Implement Deep RNNs, which have multiple layers of RNNs stacked on top of each other and can be built with any of
// the basic RNN units (vanilla, LSTM, GRU).
// 7. Implement CNN.
// 8. Work on tutorials on TensorFlow sites, e.g., Neural machine translation with a Transformer and Keras.
// 9. Implement interesting algorithms from papers.
// 10. Come up with your own algorithms and models learned from papers.
// 11. Solve and create new Mathematics and Physics problem theories using autoregressive transformer auto-generative
// type models.

int main()
{
    auto data = load_imdb();

    std::cout << "done" << std::endl;

    return 0;
}