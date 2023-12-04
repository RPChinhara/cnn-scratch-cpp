#include "datasets.h"
#include "nn.h"
#include "preprocessing.h"
#include "window.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    AllocConsole();
    freopen_s((FILE**)stdout, "CONOUT$", "w", stdout);

#if 1
    Iris iris = load_iris();
    Tensor x = iris.features;
    Tensor y = iris.target;

    y = one_hot(y, 3);
    TrainTest train_temp = train_test_split(x, y, 0.2, 42);
    TrainTest val_test = train_test_split(train_temp.x_second, train_temp.y_second, 0.5, 42);
    train_temp.x_first = min_max_scaler(train_temp.x_first);
    val_test.x_first = min_max_scaler(val_test.x_first);
    val_test.x_second = min_max_scaler(val_test.x_second);

    NN nn = NN({ 4, 128, 3 }, 0.01f);
    nn.train(train_temp.x_first, train_temp.y_first, val_test.x_first, val_test.y_first);
    nn.predict(val_test.x_second, val_test.y_second);
#endif

    try {
        Window window(hInstance, nCmdShow);
        int result = window.messageLoop();
        FreeConsole();
        return result;
    } catch (const std::exception& e) {
        MessageBox(nullptr, e.what(), "Error", MB_ICONERROR | MB_OK);
        return -1;
    }
}