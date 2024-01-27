#include "action.h"
#include "dataset.h"
#include "entity.h"
#include "environment.h"
#include "nn.h"
#include "preprocessing.h"
#include "q_learning.h"

#include <iostream>
#include <stdio.h>
#include <thread>
#include <iomanip>

#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "winmm.lib")

#define DEBUG
#define FEEDFORWARD_NEURAL_NETWORK
// #define REINFORCEMENT_LEARNING

inline std::chrono::time_point<std::chrono::high_resolution_clock> lifeStartTime;

void ResolveBoundaryCollision(RECT& rect, const LONG client_width, const LONG client_height);
void ResolveRectanglesCollision(RECT& rect1, const RECT& rect2, Entity entity, const LONG client_width, const LONG client_height);
Tensor Relu(const Tensor& in);
float CategoricalCrossEntropy(const Tensor& y_true, const Tensor& y_pred);
Tensor MatMul(const Tensor& in_1, const Tensor& in_2);
Tensor Softmax(const Tensor& in);

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
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
            HBRUSH grassBrush = CreateSolidBrush(RGB(110, 168, 88));
            FillRect(hdc, &client_rect, grassBrush);
            DeleteObject(grassBrush);

            HBRUSH whiteBrush = CreateSolidBrush(RGB(255, 255, 255));
            FillRect(hdc, &bed, whiteBrush);
            DeleteObject(whiteBrush);

            HBRUSH redBrush = CreateSolidBrush(RGB(255, 0, 0));
            FillRect(hdc, &food, redBrush);
            DeleteObject(redBrush);

            HBRUSH blueBrush = CreateSolidBrush(RGB(0, 0, 255));
            FillRect(hdc, &water, blueBrush);
            DeleteObject(blueBrush);

            HBRUSH pinkBrush = CreateSolidBrush(RGB(209, 163, 164));
            FillRect(hdc, &agent, pinkBrush);
            FillRect(hdc, &agent_2, pinkBrush);
            DeleteObject(pinkBrush);

            HBRUSH blackBrush = CreateSolidBrush(RGB(0, 0, 0));
            if (render_agent_left_eye)
                FillRect(hdc, &agent_left_eye, blackBrush);
            if (render_agent_right_eye)
                FillRect(hdc, &agent_right_eye, blackBrush);
            DeleteObject(blackBrush);

            EndPaint(hwnd, &ps);

            return 0;
        }
        default:
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    AllocConsole();
    
    FILE* file;
    freopen_s(&file, "CONOUT$", "w", stdout);

    const char CLASS_NAME[] = "WorldWindow";
    const char WINDOW_NAME[] = "Dora";

    WNDCLASS wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    if (!RegisterClass(&wc)) {
        MessageBox(nullptr, "Window Registration Failed!", "Error", MB_ICONERROR);
    }

    HWND hwnd = CreateWindow(
        CLASS_NAME,
        WINDOW_NAME,
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

#ifdef DEBUG
    Tensor a = Tensor({ -3.0, 2.0, 440.0, 2.0 }, {1, 4});
    Tensor b = Tensor({ 2, 4, 2, 2 }, { 2, 2});
    Tensor c = Tensor({ 2, 4, 2, 2 }, { 2, 2});
    std::cout << Softmax(b) << std::endl;;
    // Tensor(
    //     [[0.119203 0.880797]
    //     [0.500000 0.500000]], shape=(2, 2))
#endif

#ifdef FEEDFORWARD_NEURAL_NETWORK
    Iris iris = LoadIris();
    Tensor x = iris.features;
    Tensor y = iris.target;

    size_t depth = 3;
    float testSize1 = 0.2;
    float testSize2 = 0.5;
    size_t randomState = 42;

    y = OneHot(y, depth);
    TrainTest train_temp = TrainTestSplit(x, y, testSize1, randomState);
    TrainTest val_test = TrainTestSplit(train_temp.x_second, train_temp.y_second, testSize2, randomState);
    train_temp.x_first = MinMaxScaler(train_temp.x_first);
    val_test.x_first = MinMaxScaler(val_test.x_first);
    val_test.x_second = MinMaxScaler(val_test.x_second);

    size_t inputLayer = 4;
    size_t hiddenLayer = 128;
    size_t outputLayer = 3;
    float learningRate = 0.01f;

    NN nn = NN({ inputLayer, hiddenLayer, outputLayer }, learningRate);

    auto startTime = std::chrono::high_resolution_clock::now();
    
    nn.Train(train_temp.x_first, train_temp.y_first, val_test.x_first, val_test.y_first);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(endTime - startTime);
    std::cout << "Time taken: " << duration.count() << " seconds" << '\n';

    nn.Predict(val_test.x_second, val_test.y_second);
#endif

#ifdef REINFORCEMENT_LEARNING
    // std::thread sound_thread([]() {
    //     while (true)
    //         PlaySound(TEXT("assets\\mixkit-arcade-retro-game-over-213.wav"), NULL, SND_FILENAME);
    // });

    std::thread rl_thread([&hwnd]() {
        RECT client_rect;
        GetClientRect(hwnd, &client_rect);
        LONG client_width = client_rect.right - client_rect.left, client_height = client_rect.bottom - client_rect.top;

        agent           = { borderToAgent, (client_height - borderToAgent) - agent_height, borderToAgent + agent_width, client_height - borderToAgent };
        agent_left_eye  = { (agent.right - agentToEyeWidth) - agent_eye_width, (client_height - borderToAgent) - agent_height + agentToEyeHeight, agent.right - agentToEyeWidth, (client_height - borderToAgent) - agent_height + agentToEyeHeight + agent_eye_height };
        agent_right_eye = { agent.left + agentToEyeWidth, (client_height - borderToAgent) - agent_height + agentToEyeHeight, (agent.left + agentToEyeWidth) + agent_eye_width, (client_height - borderToAgent) - agent_height + agentToEyeHeight + agent_eye_height };
        agent_2         = { (client_width - borderToEntities) - agent_width, (client_height - borderToEntities) - agent_height, client_width - borderToEntities, client_height - borderToEntities };
        food            = { borderToEntities, borderToEntities, borderToEntities + food_width, borderToEntities + food_height };
        water           = { (client_width - borderToEntities) - water_width, borderToEntities, client_width - borderToEntities, borderToEntities + water_height };
        bed             = { borderToEntities, (client_height - borderToEntities) - bed_height, borderToEntities + bed_width, client_height - borderToEntities };
        
        Orientation orientation = Orientation::FRONT;
        Direction direction = Direction::SOUTH;

        Environment env = Environment(client_width, client_height);
        QLearning q_learning = QLearning(env.numStates, env.numActions);

        size_t num_episodes = 1000;

        for (size_t i = 0; i < num_episodes; ++i) {
            lifeStartTime = std::chrono::high_resolution_clock::now();
            auto state = env.Reset();
            bool done = false;
            float total_reward = 0;
            size_t iteration = 0;

            while (!done) {
                Action action = q_learning.ChooseAction(state);

                auto FrontConfig = [&]() {
                    orientation = Orientation::FRONT;
                    direction = Direction::SOUTH;
                    render_agent_left_eye = true;
                    render_agent_right_eye = true;
                };

                auto LeftConfig = [&]() {
                    orientation = Orientation::LEFT;
                    direction = Direction::WEST;
                    render_agent_left_eye = true;
                    render_agent_right_eye = false;
                };

                auto RightConfig = [&]() {
                    orientation = Orientation::RIGHT;
                    direction = Direction::EAST;
                    render_agent_left_eye = false;
                    render_agent_right_eye = true;
                };

                auto BackConfig = [&]() {
                    orientation = Orientation::BACK;
                    direction = Direction::NORTH;
                    render_agent_left_eye = false;
                    render_agent_right_eye = false;
                };

                size_t pixelChangeWalk = 21;
                size_t pixelChangeRun = 60;

                agent_previous = agent;

                switch (action) {
                    case Action::WALK:
                        switch (orientation) {
                            case Orientation::FRONT:
                                agent.top += pixelChangeWalk, agent.bottom += pixelChangeWalk;
                                agent_left_eye.top += pixelChangeWalk, agent_left_eye.bottom += pixelChangeWalk;
                                agent_right_eye.top += pixelChangeWalk, agent_right_eye.bottom += pixelChangeWalk;
                                break;
                            case Orientation::LEFT:
                                agent.left += pixelChangeWalk, agent.right += pixelChangeWalk;
                                agent_left_eye.left += pixelChangeWalk, agent_left_eye.right += pixelChangeWalk;
                                agent_right_eye.left += pixelChangeWalk, agent_right_eye.right += pixelChangeWalk;
                                break;
                            case Orientation::RIGHT:
                                agent.left -= pixelChangeWalk, agent.right -= pixelChangeWalk;
                                agent_left_eye.left -= pixelChangeWalk, agent_left_eye.right -= pixelChangeWalk;
                                agent_right_eye.left -= pixelChangeWalk, agent_right_eye.right -= pixelChangeWalk;
                                break;
                            case Orientation::BACK:
                                agent.top -= pixelChangeWalk, agent.bottom -= pixelChangeWalk;
                                agent_left_eye.top -= pixelChangeWalk, agent_left_eye.bottom -= pixelChangeWalk;
                                agent_right_eye.top -= pixelChangeWalk, agent_right_eye.bottom -= pixelChangeWalk;
                                break;
                            default:
                                MessageBox(nullptr, "Unknown orientation", "Error", MB_ICONERROR);
                                break;
                        }
                        break;
                    case Action::RUN:
                        switch (orientation) {
                            case Orientation::FRONT:
                                agent.top += pixelChangeRun, agent.bottom += pixelChangeRun;
                                agent_left_eye.top += pixelChangeRun, agent_left_eye.bottom += pixelChangeRun;
                                agent_right_eye.top += pixelChangeRun, agent_right_eye.bottom += pixelChangeRun;
                                break;
                            case Orientation::LEFT:
                                agent.left += pixelChangeRun, agent.right += pixelChangeRun;
                                agent_left_eye.left += pixelChangeRun, agent_left_eye.right += pixelChangeRun;
                                agent_right_eye.left += pixelChangeRun, agent_right_eye.right += pixelChangeRun;
                                break;
                            case Orientation::RIGHT:
                                agent.left -= pixelChangeRun, agent.right -= pixelChangeRun;
                                agent_left_eye.left -= pixelChangeRun, agent_left_eye.right -= pixelChangeRun;
                                agent_right_eye.left -= pixelChangeRun, agent_right_eye.right -= pixelChangeRun;
                                break;
                            case Orientation::BACK:
                                agent.top -= pixelChangeRun, agent.bottom -= pixelChangeRun;
                                agent_left_eye.top -= pixelChangeRun, agent_left_eye.bottom -= pixelChangeRun;
                                agent_right_eye.top -= pixelChangeRun, agent_right_eye.bottom -= pixelChangeRun;
                                break;
                            default:
                                MessageBox(nullptr, "Unknown orientation", "Error", MB_ICONERROR);
                                break;
                        }
                        break;
                    case Action::TURN_LEFT:
                        switch (orientation) {
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
                        switch (orientation) {
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
                    case Action::TURN_AROUND:
                        switch (orientation) {
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
                    case Action::STATIC:
                        break;
                    default:
                        MessageBox(nullptr, "Unknown action", "Error", MB_ICONERROR);
                        break;
                }

                has_collided_with_agent_2 = false;
                has_collided_with_food = false;
                has_collided_with_water = false;
                has_collided_with_wall = false;

                ResolveRectanglesCollision(agent, agent_2, Entity::AGENT2, client_width, client_height);
                ResolveRectanglesCollision(agent, food, Entity::FOOD, client_width, client_height);
                ResolveRectanglesCollision(agent, water, Entity::WATER, client_width, client_height);
                ResolveBoundaryCollision(agent, client_width, client_height);
                
                if (has_collided_with_food)
                    PlaySound(TEXT("assets\\eating_sound_effect.wav"), NULL, SND_FILENAME);
                if (has_collided_with_water)
                    PlaySound(TEXT("assets\\gulp-37759.wav"), NULL, SND_FILENAME);

                ++iteration;
                env.Render(iteration, action, q_learning.exploration_rate, direction);
                auto [next_state, reward, temp_done] = env.Step(action);
                done = temp_done;

                q_learning.UpdateQtable(state, action, reward, next_state, done);

                total_reward += reward;
                state = next_state;

                InvalidateRect(hwnd, nullptr, TRUE);

                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            std::cout << "Episode " << i + 1 << ": Total Reward = " << total_reward << "\n\n";
        }
    });

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

    FreeConsole();
    fclose(file);

    return 0;
}