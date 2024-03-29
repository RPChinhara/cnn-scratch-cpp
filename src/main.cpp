#include "activations.h"
#include "arrays.h"
#include "datasets\engspa.h"
#include "datasets\imdb.h"
#include "datasets\iris.h"
#include "datasets\mnist.h"
#include "datasets\tripadvisor.h"
#include "models\cnn2d.h"
#include "models\nn.h"
#include "models\transformer.h"
#include "preprocessing.h"
#include "random.h"

#include <memory>
#include <thread>
#include <windows.h>

#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")

LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    AllocConsole();

    FILE *file;
    freopen_s(&file, "CONOUT$", "w", stdout);

#if 0
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
#endif

#if 0
    Iris iris = LoadIris();
    Tensor features = iris.features;
    Tensor targets = iris.targets;

    targets = OneHot(targets, 3);

    TrainTest train_temp = TrainTestSplit(features, targets, 0.2, 42);
    TrainTest val_test = TrainTestSplit(train_temp.testFeatures, train_temp.testTargets, 0.5, 42);

    train_temp.trainFeatures = MinMaxScaler(train_temp.trainFeatures);
    val_test.trainFeatures = MinMaxScaler(val_test.trainFeatures);
    val_test.testFeatures = MinMaxScaler(val_test.testFeatures);

    NN nn = NN({4, 128, 3}, 0.01f);

    auto start = std::chrono::high_resolution_clock::now();

    nn.Train(train_temp.trainFeatures, train_temp.trainTargets, val_test.trainFeatures, val_test.trainTargets);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    std::cout << "Time taken: " << duration.count() << " seconds\n";

    nn.Predict(val_test.testFeatures, val_test.testTargets);
#endif

#if 0
    Tripadvisor tripadvisor = LoadTripadvisor();

    Transformer transformer = Transformer();
#endif

#if 1
    SetConsoleOutputCP(CP_UTF8);

    EngSpa engSpa = LoadEngSpa();

    for (int i = 0; i < engSpa.targetRaw.size(); ++i)
    {
        std::cout << engSpa.targetRaw[i] << std::endl;
        std::cout << engSpa.contextRaw[i] << std::endl;
    }

    Transformer transformer = Transformer();
#endif

#if 0
    IMDB imdb = LoadIMDB();

    Transformer transformer = Transformer();
#endif

    const char CLASS_NAME[] = "WindowClass";
    const char WINDOW_NAME[] = "Dora";

    WNDCLASS wc = {};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    if (!RegisterClass(&wc))
    {
        MessageBox(nullptr, "Window Registration Failed!", "Error!", MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }

    HWND hwnd = CreateWindow(CLASS_NAME, WINDOW_NAME, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 800, 600,
                             nullptr, nullptr, hInstance, nullptr);

    if (hwnd == nullptr)
    {
        MessageBox(nullptr, "Window Creation Failed!", "Error", MB_ICONERROR);
    }
    else
    {
        ShowWindow(hwnd, nCmdShow);
        UpdateWindow(hwnd);
    }

    MSG msg = {};
    while (true)
    {

        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);

            if (msg.message == WM_QUIT)
                return static_cast<int>(msg.wParam);
        }
    }

    fclose(file);
    FreeConsole();

    return static_cast<int>(msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)
    {
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    case WM_PAINT: {

        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        RECT client_rect;
        GetClientRect(hwnd, &client_rect);
        HBRUSH whiteBrush = CreateSolidBrush(RGB(255, 255, 255));
        FillRect(hdc, &client_rect, whiteBrush);
        DeleteObject(whiteBrush);

        // TextOut(hdc, 10, 10, "Hello, Windows!", 15);

        EndPaint(hwnd, &ps);
        break;
    }
    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
    return 0;
}