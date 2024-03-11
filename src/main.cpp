#include "action.h"
#include "activation.h"
#include "array.h"
#include "dataset.h"
#include "entity.h"
#include "environment.h"
#include "model.h"
#include "physics.h"
#include "preprocessing.h"
#include "random.h"

#include <memory>
#include <thread>

#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "winmm.lib")

static constexpr UINT WM_UPDATE_DISPLAY = WM_USER + 1;

struct WinData
{
    WinData() = default;
    Agent agent;
    Agent2 agent2;
    Bed bed;
    Food food;
    Water water;
};

// struct WinData;
// WinData *winData = nullptr;

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)
    {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    case WM_PAINT: {
        RECT client_rect;
        PAINTSTRUCT ps;

        WinData *winData = reinterpret_cast<WinData *>(GetWindowLongPtr(hwnd, GWLP_USERDATA));

        HDC hdc = BeginPaint(hwnd, &ps);

        GetClientRect(hwnd, &client_rect);
        HBRUSH grassBrush = CreateSolidBrush(RGB(110, 168, 88));
        FillRect(hdc, &client_rect, grassBrush);
        DeleteObject(grassBrush);

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
        FillRect(hdc, &winData->agent2.position, pinkBrush);
        DeleteObject(pinkBrush);

        HBRUSH blackBrush = CreateSolidBrush(RGB(0, 0, 0));
        if (winData->agent.render_agent_left_eye)
            FillRect(hdc, &winData->agent.leftEyePosition, blackBrush);
        if (winData->agent.render_agent_right_eye)
            FillRect(hdc, &winData->agent.rightEyePosition, blackBrush);
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

    constexpr LONG borderToAgent = 13;
    constexpr LONG borderToEntities = 5;

    WinData *winData = new WinData;
    winData->agent = Agent(client_width, client_height, borderToAgent);
    winData->agent2 = Agent2(client_width, client_height, borderToEntities);
    winData->bed = Bed(client_height, borderToEntities);
    winData->food = Food(borderToEntities);
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

    // std::thread sound_thread([]() {
    //     while (true)
    //         PlaySound(TEXT("assets\\mixkit-arcade-retro-game-over-213.wav"), NULL, SND_FILENAME);
    // });

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
                        winData->agent.position.top += pixelChangeRun, winData->agent.position.bottom += pixelChangeRun;
                        winData->agent.leftEyePosition.top += pixelChangeRun,
                            winData->agent.leftEyePosition.bottom += pixelChangeRun;
                        winData->agent.rightEyePosition.top += pixelChangeRun,
                            winData->agent.rightEyePosition.bottom += pixelChangeRun;
                        break;
                    case Orientation::LEFT:
                        winData->agent.position.left += pixelChangeRun, winData->agent.position.right += pixelChangeRun;
                        winData->agent.leftEyePosition.left += pixelChangeRun,
                            winData->agent.leftEyePosition.right += pixelChangeRun;
                        winData->agent.rightEyePosition.left += pixelChangeRun,
                            winData->agent.rightEyePosition.right += pixelChangeRun;
                        break;
                    case Orientation::RIGHT:
                        winData->agent.position.left -= pixelChangeRun, winData->agent.position.right -= pixelChangeRun;
                        winData->agent.leftEyePosition.left -= pixelChangeRun,
                            winData->agent.leftEyePosition.right -= pixelChangeRun;
                        winData->agent.rightEyePosition.left -= pixelChangeRun,
                            winData->agent.rightEyePosition.right -= pixelChangeRun;
                        break;
                    case Orientation::BACK:
                        winData->agent.position.top -= pixelChangeRun, winData->agent.position.bottom -= pixelChangeRun;
                        winData->agent.leftEyePosition.top -= pixelChangeRun,
                            winData->agent.leftEyePosition.bottom -= pixelChangeRun;
                        winData->agent.rightEyePosition.top -= pixelChangeRun,
                            winData->agent.rightEyePosition.bottom -= pixelChangeRun;
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
                        winData->agent.position.top += pixelChangeWalk,
                            winData->agent.position.bottom += pixelChangeWalk;
                        winData->agent.leftEyePosition.top += pixelChangeWalk,
                            winData->agent.leftEyePosition.bottom += pixelChangeWalk;
                        winData->agent.rightEyePosition.top += pixelChangeWalk,
                            winData->agent.rightEyePosition.bottom += pixelChangeWalk;
                        break;
                    case Orientation::LEFT:
                        winData->agent.position.left += pixelChangeWalk,
                            winData->agent.position.right += pixelChangeWalk;
                        winData->agent.leftEyePosition.left += pixelChangeWalk,
                            winData->agent.leftEyePosition.right += pixelChangeWalk;
                        winData->agent.rightEyePosition.left += pixelChangeWalk,
                            winData->agent.rightEyePosition.right += pixelChangeWalk;
                        break;
                    case Orientation::RIGHT:
                        winData->agent.position.left -= pixelChangeWalk,
                            winData->agent.position.right -= pixelChangeWalk;
                        winData->agent.leftEyePosition.left -= pixelChangeWalk,
                            winData->agent.leftEyePosition.right -= pixelChangeWalk;
                        winData->agent.rightEyePosition.left -= pixelChangeWalk,
                            winData->agent.rightEyePosition.right -= pixelChangeWalk;
                        break;
                    case Orientation::BACK:
                        winData->agent.position.top -= pixelChangeWalk,
                            winData->agent.position.bottom -= pixelChangeWalk;
                        winData->agent.leftEyePosition.top -= pixelChangeWalk,
                            winData->agent.leftEyePosition.bottom -= pixelChangeWalk;
                        winData->agent.rightEyePosition.top -= pixelChangeWalk,
                            winData->agent.rightEyePosition.bottom -= pixelChangeWalk;
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

                winData->agent.has_collided_with_agent2 = false;
                winData->agent.has_collided_with_food = false;
                winData->agent.has_collided_with_water = false;
                winData->agent.has_collided_with_wall = false;

                ResolveRectanglesCollision(winData->agent, winData->agent2, client_width, client_height);
                ResolveRectanglesCollision(winData->agent, winData->food, client_width, client_height);
                ResolveRectanglesCollision(winData->agent, winData->water, client_width, client_height);
                ResolveBoundaryCollision(winData->agent, client_width, client_height);

                if (winData->agent.has_collided_with_food)
                    PlaySound(TEXT("asset\\eating_sound_effect.wav"), NULL, SND_FILENAME);

                if (winData->agent.has_collided_with_water)
                    PlaySound(TEXT("asset\\gulp-37759.wav"), NULL, SND_FILENAME);

                ++iteration;
                environment.Render(i, iteration, action, qLearning.exploration_rate, winData->agent);
                auto [next_state, reward, temp_done] =
                    environment.Step(action, winData->agent, winData->agent2, winData->food, winData->water);
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

    FreeConsole();
    fclose(file);
    delete winData;

    return 0;
}