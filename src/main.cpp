#include "action.h"
#include "activations.h"
#include "arrays.h"
#include "datasets.h"
#include "environment.h"
#include "models\cnn2d.h"
#include "models\nn.h"
#include "models\qlearning.h"
#include "models\transformer.h"
#include "physics.h"
#include "preprocessing.h"
#include "random.h"
#include "windata.h"

#include <gdiplus.h>
#include <memory>
#include <thread>

#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "winmm.lib")

LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    Gdiplus::GdiplusStartupInput gdiplusStartupInput;
    ULONG_PTR gdiplusToken;
    Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, nullptr);

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

#if 1
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

#if 1
    Transformer transformer = Transformer();
#endif

#if 1
    const char CLASS_NAME[] = "WorldWindow";
    const char WINDOW_NAME[] = "Dora";

    WNDCLASS wc = {};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    if (!RegisterClass(&wc))
        MessageBox(nullptr, "Window Registration Failed!", "Error", MB_ICONERROR);

    HWND hwnd = CreateWindow(CLASS_NAME, WINDOW_NAME, WS_OVERLAPPEDWINDOW | WS_MAXIMIZE, CW_USEDEFAULT, CW_USEDEFAULT,
                             CW_USEDEFAULT, CW_USEDEFAULT, nullptr, nullptr, hInstance, nullptr);

    RECT client_rect;
    GetClientRect(hwnd, &client_rect);
    LONG client_width = client_rect.right - client_rect.left, client_height = client_rect.bottom - client_rect.top;

    constexpr LONG borderToEntities = 5;

    WinData *winData = new WinData;
    winData->agent = Agent(client_width, client_height);
    winData->bed = Bed(client_height, borderToEntities);
    winData->building = Building(200, 200);
    winData->food = Food(borderToEntities);
    winData->mod = Mod(client_width, client_height, borderToEntities);
    winData->street = Street(client_width, client_height);
    winData->water = Water(client_width, client_height, borderToEntities);

    SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(winData));

    if (hwnd == nullptr)
    {
        MessageBox(nullptr, "Window Creation Failed!", "Error", MB_ICONERROR);
    }
    else
    {
        ShowWindow(hwnd, SW_MAXIMIZE);
        UpdateWindow(hwnd);
    }

    std::thread rl_thread([&hwnd, winData]() {
        RECT client_rect;
        GetClientRect(hwnd, &client_rect);
        LONG client_width = client_rect.right - client_rect.left, client_height = client_rect.bottom - client_rect.top;

        Environment environment = Environment(client_width, client_height, winData->agent);
        QLearning qLearning = QLearning(environment.numStates, environment.numActions);

        size_t num_episodes = 1000;

        for (size_t i = 0; i < num_episodes; ++i)
        {
            auto state = environment.Reset(winData->agent);
            bool done = false;
            float total_reward = 0;
            size_t iteration = 0;

            while (!done)
            {
                Action action = qLearning.ChooseAction(state);

                auto FrontConfig = [&]() {
                    winData->agent.orientation = Orientation::FRONT;
                    winData->agent.direction = Direction::SOUTH;
                    winData->agent.render_agent_left_eye = true;
                    winData->agent.render_agent_right_eye = true;
                };

                auto LeftConfig = [&]() {
                    winData->agent.orientation = Orientation::LEFT;
                    winData->agent.direction = Direction::EAST;
                    winData->agent.render_agent_left_eye = true;
                    winData->agent.render_agent_right_eye = false;
                };

                auto RightConfig = [&]() {
                    winData->agent.orientation = Orientation::RIGHT;
                    winData->agent.direction = Direction::WEST;
                    winData->agent.render_agent_left_eye = false;
                    winData->agent.render_agent_right_eye = true;
                };

                auto BackConfig = [&]() {
                    winData->agent.orientation = Orientation::BACK;
                    winData->agent.direction = Direction::NORTH;
                    winData->agent.render_agent_left_eye = false;
                    winData->agent.render_agent_right_eye = false;
                };

                size_t pixelChangeWalk = 21;
                size_t pixelChangeRun = 60;

                winData->agent.previousPosition = winData->agent.position;

                switch (action)
                {
                case Action::RUN:
                    switch (winData->agent.orientation)
                    {
                    case Orientation::FRONT:
                        winData->mod.position.top -= pixelChangeRun;
                        winData->mod.position.bottom -= pixelChangeRun;

                        winData->bed.position.top -= pixelChangeRun;
                        winData->bed.position.bottom -= pixelChangeRun;

                        winData->building.y -= pixelChangeRun;

                        winData->food.position.top -= pixelChangeRun;
                        winData->food.position.bottom -= pixelChangeRun;

                        winData->street.position.top -= pixelChangeRun;
                        winData->street.position.bottom -= pixelChangeRun;

                        winData->water.position.top -= pixelChangeRun;
                        winData->water.position.bottom -= pixelChangeRun;
                        break;
                    case Orientation::LEFT:
                        winData->mod.position.left -= pixelChangeRun;
                        winData->mod.position.right -= pixelChangeRun;

                        winData->bed.position.left -= pixelChangeRun;
                        winData->bed.position.right -= pixelChangeRun;

                        winData->building.x -= pixelChangeRun;

                        winData->food.position.left -= pixelChangeRun;
                        winData->food.position.right -= pixelChangeRun;

                        winData->street.position.left -= pixelChangeRun;
                        winData->street.position.right -= pixelChangeRun;

                        winData->water.position.left -= pixelChangeRun;
                        winData->water.position.right -= pixelChangeRun;
                        break;
                    case Orientation::RIGHT:
                        winData->mod.position.left += pixelChangeRun;
                        winData->mod.position.right += pixelChangeRun;

                        winData->bed.position.left += pixelChangeRun;
                        winData->bed.position.right += pixelChangeRun;

                        winData->building.x += pixelChangeRun;

                        winData->food.position.left += pixelChangeRun;
                        winData->food.position.right += pixelChangeRun;

                        winData->street.position.left += pixelChangeRun;
                        winData->street.position.right += pixelChangeRun;

                        winData->water.position.left += pixelChangeRun;
                        winData->water.position.right += pixelChangeRun;
                        break;
                    case Orientation::BACK:
                        winData->mod.position.top += pixelChangeRun;
                        winData->mod.position.bottom += pixelChangeRun;

                        winData->bed.position.top += pixelChangeRun;
                        winData->bed.position.bottom += pixelChangeRun;

                        winData->building.y += pixelChangeRun;

                        winData->food.position.top += pixelChangeRun;
                        winData->food.position.bottom += pixelChangeRun;

                        winData->street.position.top += pixelChangeRun;
                        winData->street.position.bottom += pixelChangeRun;

                        winData->water.position.top += pixelChangeRun;
                        winData->water.position.bottom += pixelChangeRun;
                        break;
                    default:
                        MessageBox(nullptr, "Unknown orientation", "Error", MB_ICONERROR);
                        break;
                    }
                    break;
                case Action::STATIC:
                    // std::this_thread::sleep_for(std::chrono::seconds(2));
                    break;
                case Action::TALK:
                    // std::this_thread::sleep_for(std::chrono::seconds(2));
                    break;
                case Action::TURN_AROUND:
                    switch (winData->agent.orientation)
                    {
                    case Orientation::FRONT:
                        BackConfig();
                        break;
                    case Orientation::LEFT:
                        RightConfig();
                        break;
                    case Orientation::RIGHT:
                        LeftConfig();
                        break;
                    case Orientation::BACK:
                        FrontConfig();
                        break;
                    default:
                        MessageBox(nullptr, "Unknown orientation", "Error", MB_ICONERROR);
                        break;
                    }
                    break;
                case Action::TURN_LEFT:
                    switch (winData->agent.orientation)
                    {
                    case Orientation::FRONT:
                        LeftConfig();
                        break;
                    case Orientation::LEFT:
                        BackConfig();
                        break;
                    case Orientation::RIGHT:
                        FrontConfig();
                        break;
                    case Orientation::BACK:
                        RightConfig();
                        break;
                    default:
                        MessageBox(nullptr, "Unknown orientation", "Error", MB_ICONERROR);
                        break;
                    }
                    break;
                case Action::TURN_RIGHT:
                    switch (winData->agent.orientation)
                    {
                    case Orientation::FRONT:
                        RightConfig();
                        break;
                    case Orientation::LEFT:
                        FrontConfig();
                        break;
                    case Orientation::RIGHT:
                        BackConfig();
                        break;
                    case Orientation::BACK:
                        LeftConfig();
                        break;
                    default:
                        MessageBox(nullptr, "Unknown orientation", "Error", MB_ICONERROR);
                        break;
                    }
                    break;
                case Action::WALK:
                    switch (winData->agent.orientation)
                    {
                    case Orientation::FRONT:
                        winData->mod.position.top -= pixelChangeWalk;
                        winData->mod.position.bottom -= pixelChangeWalk;

                        winData->bed.position.top -= pixelChangeWalk;
                        winData->bed.position.bottom -= pixelChangeWalk;

                        winData->building.y -= pixelChangeRun;

                        winData->food.position.top -= pixelChangeWalk;
                        winData->food.position.bottom -= pixelChangeWalk;

                        winData->street.position.top -= pixelChangeWalk;
                        winData->street.position.bottom -= pixelChangeWalk;

                        winData->water.position.top -= pixelChangeWalk;
                        winData->water.position.bottom -= pixelChangeWalk;
                        break;
                    case Orientation::LEFT:
                        winData->mod.position.left -= pixelChangeWalk;
                        winData->mod.position.right -= pixelChangeWalk;

                        winData->bed.position.left -= pixelChangeWalk;
                        winData->bed.position.right -= pixelChangeWalk;

                        winData->building.x -= pixelChangeRun;

                        winData->food.position.left -= pixelChangeWalk;
                        winData->food.position.right -= pixelChangeWalk;

                        winData->street.position.left -= pixelChangeWalk;
                        winData->street.position.right -= pixelChangeWalk;

                        winData->water.position.left -= pixelChangeWalk;
                        winData->water.position.right -= pixelChangeWalk;
                        break;
                    case Orientation::RIGHT:
                        winData->mod.position.left += pixelChangeWalk;
                        winData->mod.position.right += pixelChangeWalk;

                        winData->building.x += pixelChangeRun;

                        winData->bed.position.left += pixelChangeWalk;
                        winData->bed.position.right += pixelChangeWalk;

                        winData->food.position.left += pixelChangeWalk;
                        winData->food.position.right += pixelChangeWalk;

                        winData->street.position.left += pixelChangeWalk;
                        winData->street.position.right += pixelChangeWalk;

                        winData->water.position.left += pixelChangeWalk;
                        winData->water.position.right += pixelChangeWalk;
                        break;
                    case Orientation::BACK:
                        winData->mod.position.top += pixelChangeWalk;
                        winData->mod.position.bottom += pixelChangeWalk;

                        winData->bed.position.top += pixelChangeWalk;
                        winData->bed.position.bottom += pixelChangeWalk;

                        winData->building.y += pixelChangeRun;

                        winData->food.position.top += pixelChangeWalk;
                        winData->food.position.bottom += pixelChangeWalk;

                        winData->street.position.top += pixelChangeWalk;
                        winData->street.position.bottom += pixelChangeWalk;

                        winData->water.position.top += pixelChangeWalk;
                        winData->water.position.bottom += pixelChangeWalk;
                        break;
                    default:
                        MessageBox(nullptr, "Unknown orientation", "Error", MB_ICONERROR);
                        break;
                    }
                    break;
                default:
                    MessageBox(nullptr, "Unknown action", "Error", MB_ICONERROR);
                    break;
                }

                winData->agent.has_collided_with_mod = false;
                winData->agent.has_collided_with_food = false;
                winData->agent.has_collided_with_water = false;
                winData->agent.has_collided_with_wall = false;

                ResolveRectanglesCollision(winData->agent, winData->mod, client_width, client_height);
                ResolveRectanglesCollision(winData->agent, winData->food, client_width, client_height);
                ResolveRectanglesCollision(winData->agent, winData->water, client_width, client_height);
                ResolveBoundaryCollision(winData->agent, client_width, client_height);

                InvalidateRect(hwnd, nullptr, TRUE);

                ++iteration;

                environment.Render(i, iteration, action, qLearning.exploration_rate, winData->agent);

                auto [next_state, reward, temp_done] = environment.Step(action, winData);

                done = temp_done;

                qLearning.UpdateQtable(state, action, reward, next_state, done);

                total_reward += reward;
                state = next_state;

                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            std::cout << "Episode " << i + 1 << ": Total Reward = " << total_reward << "\n\n";
        }
    });

    rl_thread.detach();
#endif

    while (true)
    {
        MSG msg = {};

        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);

            if (msg.message == WM_QUIT)
                return static_cast<int>(msg.wParam);
        }
    }

    delete winData;
    fclose(file);
    FreeConsole();
    Gdiplus::GdiplusShutdown(gdiplusToken);

    return 0;
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)
    {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    case WM_PAINT: {
        WinData *winData = reinterpret_cast<WinData *>(GetWindowLongPtr(hwnd, GWLP_USERDATA));

        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        RECT client_rect;
        GetClientRect(hwnd, &client_rect);
        HBRUSH grassBrush = CreateSolidBrush(RGB(110, 168, 88));
        FillRect(hdc, &client_rect, grassBrush);
        DeleteObject(grassBrush);

        HBRUSH greyBrush = CreateSolidBrush(RGB(128, 128, 128));
        FillRect(hdc, &winData->street.position, greyBrush);
        DeleteObject(greyBrush);

        HBRUSH whiteBrush = CreateSolidBrush(RGB(255, 255, 255));
        FillRect(hdc, &winData->bed.position, whiteBrush);
        DeleteObject(whiteBrush);

        HBRUSH redBrush = CreateSolidBrush(RGB(255, 0, 0));
        FillRect(hdc, &winData->food.position, redBrush);
        DeleteObject(redBrush);

        HBRUSH blueBrush = CreateSolidBrush(RGB(0, 0, 255));
        FillRect(hdc, &winData->water.position, blueBrush);
        DeleteObject(blueBrush);

        HBRUSH pinkBrush = CreateSolidBrush(RGB(209, 163, 164));
        FillRect(hdc, &winData->agent.position, pinkBrush);
        FillRect(hdc, &winData->mod.position, pinkBrush);
        DeleteObject(pinkBrush);

        HBRUSH blackBrush = CreateSolidBrush(RGB(0, 0, 0));
        if (winData->agent.render_agent_left_eye)
            FillRect(hdc, &winData->agent.leftEyePosition, blackBrush);
        if (winData->agent.render_agent_right_eye)
            FillRect(hdc, &winData->agent.rightEyePosition, blackBrush);
        DeleteObject(blackBrush);

        Gdiplus::Graphics gf(hdc);
        Gdiplus::Bitmap bmp(L"assets\\textures\\13031.jpg");
        gf.DrawImage(&bmp, winData->building.x, winData->building.y);

        // TextOut(hdc, 10, 10, "Hello, Windows!", 15);

        EndPaint(hwnd, &ps);

        return 0;
    }
    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}