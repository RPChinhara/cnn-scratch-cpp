#include "datasets.h"
#include "nn.h"
#include "preprocessing.h"
#include "window.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    // Create a console for logging otherwise I can't when WinMain() is used as the entry point because it doesn't use the standard console for input and output
    AllocConsole();
    freopen_s((FILE**)stdout, "CONOUT$", "w", stdout);

    // Load the Iris dataset
    Iris iris = load_iris();
    Tensor x = iris.features;
    Tensor y = iris.target;

    // Preprocess the dataset
    y = one_hot(y, 3);
    TrainTest2 train_temp = train_test_split(x, y, 0.2, 42);
    TrainTest2 val_test   = train_test_split(train_temp.x_second, train_temp.y_second, 0.5, 42);
    train_temp.x_first = min_max_scaler(train_temp.x_first);
    val_test.x_first   = min_max_scaler(val_test.x_first);
    val_test.x_second  = min_max_scaler(val_test.x_second);

    // Train and test neural network
    NN nn = NN({ 4, 128, 3 }, 0.01f);
    nn.train(train_temp.x_first, train_temp.y_first, val_test.x_first, val_test.y_first);
    nn.predict(val_test.x_second, val_test.y_second);
    
    // Making the window
    try {
        Window window(hInstance, nCmdShow);
        return window.messageLoop();
    } catch (const std::exception& e) {
        // Handle the error, e.g., show an error message
        MessageBox(NULL, e.what(), "Error", MB_ICONERROR | MB_OK);
        return -1;
    }
}