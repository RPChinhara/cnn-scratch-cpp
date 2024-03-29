#include "models\transformer.h"
#include "activations.h"
#include "arrays.h"
#include "datasets\englishspanish.h"
#include "datasets\imdb.h"
#include "datasets\iris.h"
#include "datasets\mnist.h"
#include "datasets\tripadvisor.h"
#include "models\cnn2d.h"
#include "models\nn.h"
#include "preprocessing.h"
#include "random.h"

int main()
{
    IMDB imdb = LoadIMDB();

    Transformer transformer = Transformer();

    return 0;
}