#include "datas.h"
#include "lyrs.h"

// 1. Implement Vanilla RNNs which are simplest form of RNNs. Has only single hidden layer.
//      - Implement forward prop for many-to-one and many-to-many as these are more common. One-to-one and one-to-many
//      are less common. One-to-one might be just regular normal neural netwok when you think about it...
//      - Implement backprop for all the correspondence for above.
// 2. Implement LSTM
// 3. Implement GRU
// 4. Implement Bidirectional RNNs
// 5. Implement Deep RNNs which have multiple layers of RNNs stacked on top of each other, and can be built with any of
// the basic RNN units (vanilla, LSTM, GRU)

int main()
{
    auto data = load_imdb();

    return 0;
}