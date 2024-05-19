#include "datas.h"
#include "lyrs.h"
#include "preproc.h"

int main()
{
    MNIST mnist = LoadMNIST();

    for (auto i = 0; i < 784; ++i)
    {
        if (i % 28 == 0)
            std::cout << std::endl;
        std::cout << mnist.trainImages[i] << "   ";
    }

    mnist.trainImages / 255.0f;
    mnist.testImages / 255.0f;

    mnist.trainLabels = OneHot(mnist.trainLabels, 10);
    mnist.testLabels = OneHot(mnist.testLabels, 10);

    cnn2d cnn2D = cnn2d({3, 128, 3}, 0.01f);
    cnn2D.Train(mnist.trainImages, mnist.trainLabels, mnist.testImages, mnist.testLabels);

    return 0;
}