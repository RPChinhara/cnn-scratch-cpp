#include "models\cnn2d.h"
#include "datasets\mnist.h"
#include "preprocessing.h"

int main()
{
    MNIST mnist = LoadMNIST();

    for (size_t i = 0; i < 784; ++i)
    {

        if (i % 28 == 0)
            std::cout << std::endl;
        std::cout << mnist.trainImages[i] << "   ";
    }

    mnist.trainImages / 255.0f;
    mnist.testImages / 255.0f;

    mnist.trainLabels = OneHot(mnist.trainLabels, 10);
    mnist.testLabels = OneHot(mnist.testLabels, 10);

    CNN2D cnn2D = CNN2D({3, 128, 3}, 0.01f);
    cnn2D.Train(mnist.trainImages, mnist.trainLabels, mnist.testImages, mnist.testLabels);

    return 0;
}