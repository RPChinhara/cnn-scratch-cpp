#include "action.h"
#include "activations.h"
#include "arrays.h"
#include "datasets.h"
#include "entities.h"
#include "environment.h"
#include "models.h"
#include "physics.h"
#include "preprocessing.h"
#include "random.h"

#include <gdiplus.h>
#include <memory>
#include <thread>

#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "winmm.lib")

static constexpr UINT WM_UPDATE_DISPLAY = WM_USER + 1;

// Entities *entities = nullptr;

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

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
    wc.lpfnWndProc = WindowProc;
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

    Entities *entities = new Entities;
    entities->agent = Agent(client_width, client_height);
    entities->agent2 = Agent2(client_width, client_height, borderToEntities);
    entities->bed = Bed(client_height, borderToEntities);
    entities->food = Food(borderToEntities);
    entities->water = Water(client_width, client_height, borderToEntities);

    SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(entities));

    if (hwnd == nullptr)
    {
        MessageBox(nullptr, "Window Creation Failed!", "Error", MB_ICONERROR);
    }
    else
    {
        ShowWindow(hwnd, SW_MAXIMIZE);
        UpdateWindow(hwnd);
    }

    // std::thread sound_thread([]() {
    //     while (true)
    //         PlaySound(TEXT("assets\\mixkit-arcade-retro-game-over-213.wav"), NULL, SND_FILENAME);
    // });

    std::thread rl_thread([&hwnd, entities]() {
        RECT client_rect;
        GetClientRect(hwnd, &client_rect);
        LONG client_width = client_rect.right - client_rect.left, client_height = client_rect.bottom - client_rect.top;

        Environment environment = Environment(client_width, client_height, entities->agent);
        QLearning qLearning = QLearning(environment.numStates, environment.numActions);

        size_t num_episodes = 1000;

        for (size_t i = 0; i < num_episodes; ++i)
        {
            auto state = environment.Reset(entities->agent);
            bool done = false;
            float total_reward = 0;
            size_t iteration = 0;

            while (!done)
            {
                Action action = qLearning.ChooseAction(state);

                auto FrontConfig = [&]() {
                    entities->agent.orientation = Orientation::FRONT;
                    entities->agent.direction = Direction::SOUTH;
                    entities->agent.render_agent_left_eye = true;
                    entities->agent.render_agent_right_eye = true;
                };

                auto LeftConfig = [&]() {
                    entities->agent.orientation = Orientation::LEFT;
                    entities->agent.direction = Direction::EAST;
                    entities->agent.render_agent_left_eye = true;
                    entities->agent.render_agent_right_eye = false;
                };

                auto RightConfig = [&]() {
                    entities->agent.orientation = Orientation::RIGHT;
                    entities->agent.direction = Direction::WEST;
                    entities->agent.render_agent_left_eye = false;
                    entities->agent.render_agent_right_eye = true;
                };

                auto BackConfig = [&]() {
                    entities->agent.orientation = Orientation::BACK;
                    entities->agent.direction = Direction::NORTH;
                    entities->agent.render_agent_left_eye = false;
                    entities->agent.render_agent_right_eye = false;
                };

                size_t pixelChangeWalk = 21;
                size_t pixelChangeRun = 60;

                entities->agent.previousPosition = entities->agent.position;

                switch (action)
                {
                case Action::RUN:
                    switch (entities->agent.orientation)
                    {
                    case Orientation::FRONT:
                        entities->agent2.position.top -= pixelChangeRun;
                        entities->agent2.position.bottom -= pixelChangeRun;

                        entities->bed.position.top -= pixelChangeRun;
                        entities->bed.position.bottom -= pixelChangeRun;

                        entities->food.position.top -= pixelChangeRun;
                        entities->food.position.bottom -= pixelChangeRun;

                        entities->water.position.top -= pixelChangeRun;
                        entities->water.position.bottom -= pixelChangeRun;
                        break;
                    case Orientation::LEFT:
                        entities->agent2.position.left -= pixelChangeRun;
                        entities->agent2.position.right -= pixelChangeRun;

                        entities->bed.position.left -= pixelChangeRun;
                        entities->bed.position.right -= pixelChangeRun;

                        entities->food.position.left -= pixelChangeRun;
                        entities->food.position.right -= pixelChangeRun;

                        entities->water.position.left -= pixelChangeRun;
                        entities->water.position.right -= pixelChangeRun;
                        break;
                    case Orientation::RIGHT:
                        entities->agent2.position.left += pixelChangeRun;
                        entities->agent2.position.right += pixelChangeRun;

                        entities->bed.position.left += pixelChangeRun;
                        entities->bed.position.right += pixelChangeRun;

                        entities->food.position.left += pixelChangeRun;
                        entities->food.position.right += pixelChangeRun;

                        entities->water.position.left += pixelChangeRun;
                        entities->water.position.right += pixelChangeRun;
                        break;
                    case Orientation::BACK:
                        entities->agent2.position.top += pixelChangeRun;
                        entities->agent2.position.bottom += pixelChangeRun;

                        entities->bed.position.top += pixelChangeRun;
                        entities->bed.position.bottom += pixelChangeRun;

                        entities->food.position.top += pixelChangeRun;
                        entities->food.position.bottom += pixelChangeRun;

                        entities->water.position.top += pixelChangeRun;
                        entities->water.position.bottom += pixelChangeRun;
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
                    switch (entities->agent.orientation)
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
                    switch (entities->agent.orientation)
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
                    switch (entities->agent.orientation)
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
                    switch (entities->agent.orientation)
                    {
                    case Orientation::FRONT:
                        entities->agent2.position.top -= pixelChangeWalk;
                        entities->agent2.position.bottom -= pixelChangeWalk;

                        entities->bed.position.top -= pixelChangeWalk;
                        entities->bed.position.bottom -= pixelChangeWalk;

                        entities->food.position.top -= pixelChangeWalk;
                        entities->food.position.bottom -= pixelChangeWalk;

                        entities->water.position.top -= pixelChangeWalk;
                        entities->water.position.bottom -= pixelChangeWalk;
                        break;
                    case Orientation::LEFT:
                        entities->agent2.position.left -= pixelChangeWalk;
                        entities->agent2.position.right -= pixelChangeWalk;

                        entities->bed.position.left -= pixelChangeWalk;
                        entities->bed.position.right -= pixelChangeWalk;

                        entities->food.position.left -= pixelChangeWalk;
                        entities->food.position.right -= pixelChangeWalk;

                        entities->water.position.left -= pixelChangeWalk;
                        entities->water.position.right -= pixelChangeWalk;
                        break;
                    case Orientation::RIGHT:
                        entities->agent2.position.left += pixelChangeWalk;
                        entities->agent2.position.right += pixelChangeWalk;

                        entities->bed.position.left += pixelChangeWalk;
                        entities->bed.position.right += pixelChangeWalk;

                        entities->food.position.left += pixelChangeWalk;
                        entities->food.position.right += pixelChangeWalk;

                        entities->water.position.left += pixelChangeWalk;
                        entities->water.position.right += pixelChangeWalk;
                        break;
                    case Orientation::BACK:
                        entities->agent2.position.top += pixelChangeWalk;
                        entities->agent2.position.bottom += pixelChangeWalk;

                        entities->bed.position.top += pixelChangeWalk;
                        entities->bed.position.bottom += pixelChangeWalk;

                        entities->food.position.top += pixelChangeWalk;
                        entities->food.position.bottom += pixelChangeWalk;

                        entities->water.position.top += pixelChangeWalk;
                        entities->water.position.bottom += pixelChangeWalk;
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

                entities->agent.has_collided_with_agent2 = false;
                entities->agent.has_collided_with_food = false;
                entities->agent.has_collided_with_water = false;
                entities->agent.has_collided_with_wall = false;

                ResolveRectanglesCollision(entities->agent, entities->agent2, client_width, client_height);
                ResolveRectanglesCollision(entities->agent, entities->food, client_width, client_height);
                ResolveRectanglesCollision(entities->agent, entities->water, client_width, client_height);
                ResolveBoundaryCollision(entities->agent, client_width, client_height);

                if (entities->agent.has_collided_with_food)
                    PlaySound(TEXT("asset\\eating_sound_effect.wav"), NULL, SND_FILENAME);

                if (entities->agent.has_collided_with_water)
                    PlaySound(TEXT("asset\\gulp-37759.wav"), NULL, SND_FILENAME);

                ++iteration;
                environment.Render(i, iteration, action, qLearning.exploration_rate, entities->agent);
                auto [next_state, reward, temp_done] = environment.Step(action, *entities);
                done = temp_done;

                qLearning.UpdateQtable(state, action, reward, next_state, done);

                total_reward += reward;
                state = next_state;

                // InvalidateRect(hwnd, nullptr, TRUE);
                PostMessage(hwnd, WM_UPDATE_DISPLAY, 0, 0);

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

    delete entities;
    fclose(file);
    FreeConsole();
    Gdiplus::GdiplusShutdown(gdiplusToken);

    return 0;
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)
    {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    case WM_PAINT: {
        Entities *entities = reinterpret_cast<Entities *>(GetWindowLongPtr(hwnd, GWLP_USERDATA));

        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        RECT client_rect;
        GetClientRect(hwnd, &client_rect);
        HBRUSH grassBrush = CreateSolidBrush(RGB(110, 168, 88));
        FillRect(hdc, &client_rect, grassBrush);
        DeleteObject(grassBrush);

        HBRUSH whiteBrush = CreateSolidBrush(RGB(255, 255, 255));
        FillRect(hdc, &entities->bed.position, whiteBrush);
        DeleteObject(whiteBrush);

        HBRUSH redBrush = CreateSolidBrush(RGB(255, 0, 0));
        FillRect(hdc, &entities->food.position, redBrush);
        DeleteObject(redBrush);

        HBRUSH blueBrush = CreateSolidBrush(RGB(0, 0, 255));
        FillRect(hdc, &entities->water.position, blueBrush);
        DeleteObject(blueBrush);

        HBRUSH pinkBrush = CreateSolidBrush(RGB(209, 163, 164));
        FillRect(hdc, &entities->agent.position, pinkBrush);
        FillRect(hdc, &entities->agent2.position, pinkBrush);
        DeleteObject(pinkBrush);

        HBRUSH blackBrush = CreateSolidBrush(RGB(0, 0, 0));
        if (entities->agent.render_agent_left_eye)
            FillRect(hdc, &entities->agent.leftEyePosition, blackBrush);
        if (entities->agent.render_agent_right_eye)
            FillRect(hdc, &entities->agent.rightEyePosition, blackBrush);
        DeleteObject(blackBrush);

        // TextOut(hdc, 10, 10, "Hello, Windows!", 15);

        EndPaint(hwnd, &ps);

        return 0;
    }
    case WM_UPDATE_DISPLAY:
        InvalidateRect(hwnd, nullptr, TRUE);
        return 0;
    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}