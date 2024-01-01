#include "window.h"
#include "dataset.h"
#include "entity.h"
#include "environment.h"
#include "nn.h"
#include "physics.h"
#include "preprocessing.h"
#include "q_learning.h"

#include <iostream>
#include <thread>

#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "winmm.lib")

static constexpr UINT WM_UPDATE_DISPLAY = WM_USER + 1;

const char Window::CLASS_NAME[] = "WorldWindow";

inline std::chrono::time_point<std::chrono::high_resolution_clock> lifeStartTime;

Window::Window(HINSTANCE hInst, int nCmdShow) : hInstance(hInst), hwnd(nullptr)
{
    WNDCLASS wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    if (!RegisterClass(&wc)) {
        MessageBox(nullptr, "Window Registration Failed!", "Error", MB_ICONERROR);
    }

    hwnd = CreateWindow(
        CLASS_NAME,
        "Dora",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        nullptr,
        nullptr,
        hInstance,
        nullptr
    );

    if (hwnd == nullptr) {
        MessageBox(nullptr, "Window Creation Failed!", "Error", MB_ICONERROR);
    } else {
        ShowWindow(hwnd, SW_MAXIMIZE);
        UpdateWindow(hwnd);
    }
}

int Window::MessageLoop()
{
#if 0
    Iris iris = LoadIris();
    Tensor x = iris.features;
    Tensor y = iris.target;

    y = OneHot(y, 3);
    TrainTest train_temp = TrainTestSplit(x, y, 0.2, 42);
    TrainTest val_test = TrainTestSplit(train_temp.x_second, train_temp.y_second, 0.5, 42);
    train_temp.x_first = MinMaxScaler(train_temp.x_first);
    val_test.x_first = MinMaxScaler(val_test.x_first);
    val_test.x_second = MinMaxScaler(val_test.x_second);

    NN nn = NN({ 4, 128, 3 }, 0.01f);

    auto startTime = std::chrono::high_resolution_clock::now();
    
    nn.Train(train_temp.x_first, train_temp.y_first, val_test.x_first, val_test.y_first);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(endTime - startTime);
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    nn.Predict(val_test.x_second, val_test.y_second);
#endif

#if 1
    std::thread sound_thread([this]() {
        while (true)
            PlaySound(TEXT("assets\\mixkit-city-traffic-background-ambience-2930.wav"), NULL, SND_FILENAME);
    });

    std::thread rl_thread([this]() {
        RECT client_rect;
        GetClientRect(hwnd, &client_rect);
        LONG client_width = client_rect.right - client_rect.left, client_height = client_rect.bottom - client_rect.top;

        agent       = { 13, (client_height - 13) - agent_height, 13 + agent_width, client_height - 13 };
        agent_eye_1 = { 23, (client_height - 13) - agent_height + 10, 23 + agent_eye_width, (client_height - 13) - agent_height + 10 + agent_eye_height };
        agent_eye_2 = { 53 - agent_eye_width, (client_height - 13) - agent_height + 10, 53, (client_height - 13) - agent_height + 10 + agent_eye_height };
        agent_2     = { (client_width - 5) - agent_width, (client_height - 5) - agent_height, client_width - 5, client_height - 5 };
        food        = { 5, 5, 5 + food_width, 5 + food_height };
        water       = { (client_width - 5) - water_width, 5, client_width - 5, 5 + water_height };
        bed         = { 5, (client_height - 5) - bed_height, 5 + bed_width, client_height - 5 };

        Environment env = Environment();
        QLearning q_learning = QLearning(env.numStates, env.numActions);

        size_t num_episodes = 1000;
        Orientation orientation = Orientation::FRONT;

        for (size_t i = 0; i < num_episodes; ++i) {
            lifeStartTime = std::chrono::high_resolution_clock::now();
            auto state = env.Reset();
            bool done = false;
            int total_reward = 0;

            while (!done) {
                size_t action = q_learning.ChooseAction(state);

                switch (action) {
                    case Action::MOVE_FORWARD: {
                        switch (orientation) {
                            case Orientation::FRONT: {
                                agent.top += 1, agent.bottom += 1;
                                agent_eye_1.top += 1, agent_eye_1.bottom += 1;
                                agent_eye_2.top += 1, agent_eye_2.bottom += 1;
                                std::cout << "fuck you " << std::endl;
                                break;
                            }
                            case Orientation::LEFT:
                                break;
                            case Orientation::RIGHT:
                                break;
                            case Orientation::BACK:
                                break;
                            default:
                                MessageBox(nullptr, "Unknown orientation", "Error", MB_ICONERROR);
                                break;
                        }
                        break;
                    }
                    case Action::TURN_LEFT:
                        break;
                    case Action::TURN_RIGHT:
                        break;
                    case Action::TURN_AROUND:
                        break;
                    case Action::STATIC:
                        break;
                    default:
                        MessageBox(nullptr, "Unknown action", "Error", MB_ICONERROR);
                        break;
                }

                // if (action == Action::MOVE_FORWARD)
                //     agent.top -= 1, agent.bottom -= 1;
                // else if (action == Action::MOVE_DOWN)
                //     agent.top += 1, agent.bottom += 1;
                // else if (action == Action::MOVE_LEFT)
                //     agent.left -= 1, agent.right -= 1;
                // else if (action == Action::MOVE_RIGHT)
                //     agent.left += 1, agent.right += 1;

                has_collided_with_agent_2 = false;
                has_collided_with_food = false;
                has_collided_with_water = false;
                has_collided_with_wall = false;

                ResolveBoundaryCollision(agent, client_width, client_height);
                ResolveRectanglesCollision(agent, agent_2, Entity::AGENT2);
                ResolveRectanglesCollision(agent, food, Entity::FOOD);
                ResolveRectanglesCollision(agent, water, Entity::WATER);

                auto [next_state, reward, temp_done] = env.Step(action);
                done = temp_done;

                q_learning.UpdateQtable(state, action, reward, next_state, done);

                total_reward += reward;
                state = next_state;

                PostMessage(hwnd, WM_UPDATE_DISPLAY, 0, 0);
                // InvalidateRect(hwnd, nullptr, TRUE);

                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                // Sleep(1000);
            }

            std::cout << "Episode " << i + 1 << ": Total Reward = " << total_reward << std::endl << std::endl;
        }
    });

    sound_thread.detach();
    rl_thread.detach();
#endif

    while(true) {
        MSG msg = {};

        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);

            if (msg.message == WM_QUIT)
                return static_cast<int>(msg.wParam);
        }
    }

    return 0;
}

LRESULT CALLBACK Window::WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg) {
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        case WM_PAINT: {
            RECT client_rect;
            PAINTSTRUCT ps;

            HDC hdc = BeginPaint(hwnd, &ps);

            GetClientRect(hwnd, &client_rect);
            HBRUSH greenBrush = CreateSolidBrush(RGB(34, 139, 34));
            FillRect(hdc, &client_rect, greenBrush);
            DeleteObject(greenBrush);

            HBRUSH whiteBrush = CreateSolidBrush(RGB(255, 255, 255));
            FillRect(hdc, &bed, whiteBrush);
            DeleteObject(whiteBrush);

            HBRUSH pinkBrush = CreateSolidBrush(RGB(209, 163, 164));
            FillRect(hdc, &agent, pinkBrush);
            FillRect(hdc, &agent_2, pinkBrush);
            DeleteObject(pinkBrush);

            HBRUSH blackBrush = CreateSolidBrush(RGB(0, 0, 0));
            FillRect(hdc, &agent_eye_1, blackBrush);
            FillRect(hdc, &agent_eye_2, blackBrush);
            DeleteObject(blackBrush);

            HBRUSH redBrush = CreateSolidBrush(RGB(255, 0, 0));
            FillRect(hdc, &food, redBrush);
            DeleteObject(redBrush);

            HBRUSH blueBrush = CreateSolidBrush(RGB(0, 0, 255));
            FillRect(hdc, &water, blueBrush);
            DeleteObject(blueBrush);

            EndPaint(hwnd, &ps);

            return 0;
        }
        case WM_UPDATE_DISPLAY: {
            InvalidateRect(hwnd, nullptr, TRUE);
            return 0;
        }
        default:
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}