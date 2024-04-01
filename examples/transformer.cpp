#include "models\transformer.h"
#include "datasets\imdb.h"
#include "preproc.h"

int main()
{
    IMDB imdb = LoadIMDB();

    Transformer transformer = Transformer();

    return 0;
}