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

// from tensorflow
// this film was just brilliant casting location scenery story direction everyone's really suited the
// part they played and you could just imagine being there robert ? is an amazing actor and now the same being director
// ? father came from the same scottish island as myself so i loved the fact there was a real connection with this film
// the witty remarks throughout the film were great it was just brilliant so much that i bought the film as soon as it
// was released for ? and would recommend it to everyone to watch and the fly fishing was amazing really cried at the
// end it was so sad and you know what they say if you cry at a film it must have been good and this definitely was also
// ? to the two little boy's that played the ? of norman and paul they were just brilliant children are often left out
// of the ? list i think because the stars that play them all grown up are such a big profile for the whole film but
// these children are amazing and should be praised for what they have done don't you think the whole story was so
// lovely because it was true and was someone's life after all that was shared with us all

// original
// this film was just brilliant,casting,location scenery,story,direction,everyone's really suited the part they
// played,and you could just imagine being there,Robert Redford's is an amazing actor and now the same being
// director,Norman's father came from the same Scottish island as myself,so i loved the fact there was a real connection
// with this film,the witty remarks throughout the film were great,it was just brilliant,so much that i bought the film
// as soon as it was released for retail and would recommend it to everyone to watch,and the fly-fishing was
// amazing,really cried at the end it was so sad,and you know what they say if you cry at a film it must have been
// good,and this definitely was, also congratulations to the two little boy's that played the part's of Norman and Paul
// they were just brilliant,children are often left out of the praising list i think, because the stars that play them
// all grown up are such a big profile for the whole film,but these children are amazing and should be praised for what
// they have done, don't you think? the whole story was so lovely because it was true and was someone's life after all
// that was shared with us all.
