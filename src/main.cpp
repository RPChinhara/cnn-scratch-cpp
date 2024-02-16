#include "action.h"
#include "activation.h"
#include "agent.h"
#include "dataset.h"
#include "entity.h"
#include "environment.h"
#include "nn.h"
#include "physics.h"
#include "preprocessing.h"
#include "q_learning.h"

#include <cstdint>
#include <cuda_fp16.h>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <thread>
#include <type_traits>

#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "winmm.lib")

static constexpr UINT WM_UPDATE_DISPLAY = WM_USER + 1;

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

        // Retrieve the pointer to your data from the window
        LONG_PTR userData = GetWindowLongPtr(hwnd, GWLP_USERDATA);

        // Assuming your data is a pointer to an int
        Agent *agent = reinterpret_cast<Agent *>(userData);

        // // Access the variable
        // int result = *myVariable;

        HBRUSH pinkBrush = CreateSolidBrush(RGB(209, 163, 164));
        FillRect(hdc, &agent->position, pinkBrush);
        FillRect(hdc, &agent2, pinkBrush);
        DeleteObject(pinkBrush);

        HBRUSH blackBrush = CreateSolidBrush(RGB(0, 0, 0));
        if (agent->render_agent_left_eye)
            FillRect(hdc, &agent->leftEyePosition, blackBrush);
        if (agent->render_agent_right_eye)
            FillRect(hdc, &agent->rightEyePosition, blackBrush);
        DeleteObject(blackBrush);

        HBRUSH brownBrush = CreateSolidBrush(RGB(165, 42, 42));
        FillRect(hdc, &predator, brownBrush);
        DeleteObject(brownBrush);

        // Draw "Hello, Windows!" text
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

#if 1
#endif

#if 0
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
    size_t hiddenLayer1 = 128;
    size_t outputLayer = 3;
    float learningRate = 0.01f;

    NN nn = NN({inputLayer, hiddenLayer1, outputLayer}, learningRate);

    auto startTime = std::chrono::high_resolution_clock::now();

    nn.Train(train_temp.x_first, train_temp.y_first, val_test.x_first, val_test.y_first);

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(endTime - startTime);
    std::cout << "Time taken: " << duration.count() << " seconds" << '\n';

    nn.Predict(val_test.x_second, val_test.y_second);
#endif

    // TODO:  Spawn stick in the env, and maybe he can pick up that. Add inventroy box he can open?
    // TODO:  No more static water, food, agent2, and predator spawn them in the random spaces.
    // TODO:  I must build from CMake
    // TODO:  He can sleep anywhere he want, but he might get eaten by predator so in order to prevent that he has to
    // build a house, and also able to make a fire reference MineCraft to how things are working in this game.
    // TODO:  Instead of food place some animal like a sheep, and he can’t eat untill he kills it, and cook that
    // TODO:  Make inventory
    // TODO:  Make trees
    // TODO:  Reference Terraria and Stardew Valley as well
    // TODO:  How game engine is implemented? Can I reference this? Is onRender() or other on..() famous in game engine?

#if 1
    const char CLASS_NAME[] = "WorldWindow";
    const char WINDOW_NAME[] = "Dora";

    WNDCLASS wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    if (!RegisterClass(&wc))
    {
        MessageBox(nullptr, "Window Registration Failed!", "Error", MB_ICONERROR);
    }

    HWND hwnd = CreateWindow(CLASS_NAME, WINDOW_NAME, WS_OVERLAPPEDWINDOW | WS_MAXIMIZE, CW_USEDEFAULT, CW_USEDEFAULT,
                             CW_USEDEFAULT, CW_USEDEFAULT, nullptr, nullptr, hInstance, nullptr);

    RECT client_rect;
    GetClientRect(hwnd, &client_rect);
    LONG client_width = client_rect.right - client_rect.left, client_height = client_rect.bottom - client_rect.top;

    Agent agent;

    agent.position = {borderToAgent, (client_height - borderToAgent) - agent.height, borderToAgent + agent.width,
                      client_height - borderToAgent};

    agent.leftEyePosition = {(agent.position.right - agent.toEyeWidth) - agent.eye_width,
                             (client_height - borderToAgent) - agent.height + agent.toEyeHeight,
                             agent.position.right - agent.toEyeWidth,
                             (client_height - borderToAgent) - agent.height + agent.toEyeHeight + agent.eye_height};

    agent.rightEyePosition = {agent.position.left + agent.toEyeWidth,
                              (client_height - borderToAgent) - agent.height + agent.toEyeHeight,
                              (agent.position.left + agent.toEyeWidth) + agent.eye_width,
                              (client_height - borderToAgent) - agent.height + agent.toEyeHeight + agent.eye_height};

    agent2 = {(client_width - borderToEntities) - agent.width, (client_height - borderToEntities) - agent.height,
              client_width - borderToEntities, client_height - borderToEntities};

    food = {borderToEntities, borderToEntities, borderToEntities + food_width, borderToEntities + food_height};

    water = {(client_width - borderToEntities) - water_width, borderToEntities, client_width - borderToEntities,
             borderToEntities + water_height};

    predator = {(client_width - borderToEntities) - predator_width, 500, client_width - borderToEntities,
                500 + predator_height};

    bed = {borderToEntities, (client_height - borderToEntities) - bed_height, borderToEntities + bed_width,
           client_height - borderToEntities};

    SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(&agent));

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

    std::thread rl_thread([&hwnd, &agent]() {
        RECT client_rect;
        GetClientRect(hwnd, &client_rect);
        LONG client_width = client_rect.right - client_rect.left, client_height = client_rect.bottom - client_rect.top;

        Orientation orientation = Orientation::FRONT;
        Direction direction = Direction::SOUTH;

        // IDEA:  Write total reward and num episodes in file so that I can leave the program while it's running and see how things gonna work.
        Environment env = Environment(client_width, client_height, agent);
        QLearning q_learning = QLearning(env.numStates, env.numActions); // TODO:  Use deep learning/neural network RL.

        size_t num_episodes = 1000;

        for (size_t i = 0; i < num_episodes; ++i)
        {
            auto state = env.Reset(agent);
            bool done = false;
            float total_reward = 0;
            size_t iteration = 0;

            while (!done)
            {
                Action action = q_learning.ChooseAction(state);

                auto FrontConfig = [&]() {
                    orientation = Orientation::FRONT;
                    direction = Direction::SOUTH;
                    agent.render_agent_left_eye = true;
                    agent.render_agent_right_eye = true;
                };

                auto LeftConfig = [&]() {
                    orientation = Orientation::LEFT;
                    direction = Direction::WEST;
                    agent.render_agent_left_eye = true;
                    agent.render_agent_right_eye = false;
                };

                auto RightConfig = [&]() {
                    orientation = Orientation::RIGHT;
                    direction = Direction::EAST;
                    agent.render_agent_left_eye = false;
                    agent.render_agent_right_eye = true;
                };

                auto BackConfig = [&]() {
                    orientation = Orientation::BACK;
                    direction = Direction::NORTH;
                    agent.render_agent_left_eye = false;
                    agent.render_agent_right_eye = false;
                };

                size_t pixelChangeWalk = 21;
                size_t pixelChangeRun = 60;

                agent.previousPosition = agent.position;

                switch (action)
                {
                case Action::WALK:
                    switch (orientation)
                    {
                    case Orientation::FRONT:
                        agent.position.top += pixelChangeWalk, agent.position.bottom += pixelChangeWalk;
                        agent.leftEyePosition.top += pixelChangeWalk, agent.leftEyePosition.bottom += pixelChangeWalk;
                        agent.rightEyePosition.top += pixelChangeWalk, agent.rightEyePosition.bottom += pixelChangeWalk;
                        break;
                    case Orientation::LEFT:
                        agent.position.left += pixelChangeWalk, agent.position.right += pixelChangeWalk;
                        agent.leftEyePosition.left += pixelChangeWalk, agent.leftEyePosition.right += pixelChangeWalk;
                        agent.rightEyePosition.left += pixelChangeWalk, agent.rightEyePosition.right += pixelChangeWalk;
                        break;
                    case Orientation::RIGHT:
                        agent.position.left -= pixelChangeWalk, agent.position.right -= pixelChangeWalk;
                        agent.leftEyePosition.left -= pixelChangeWalk, agent.leftEyePosition.right -= pixelChangeWalk;
                        agent.rightEyePosition.left -= pixelChangeWalk, agent.rightEyePosition.right -= pixelChangeWalk;
                        break;
                    case Orientation::BACK:
                        agent.position.top -= pixelChangeWalk, agent.position.bottom -= pixelChangeWalk;
                        agent.leftEyePosition.top -= pixelChangeWalk, agent.leftEyePosition.bottom -= pixelChangeWalk;
                        agent.rightEyePosition.top -= pixelChangeWalk, agent.rightEyePosition.bottom -= pixelChangeWalk;
                        break;
                    default:
                        MessageBox(nullptr, "Unknown orientation", "Error", MB_ICONERROR);
                        break;
                    }
                    break;
                case Action::RUN:
                    switch (orientation)
                    {
                    case Orientation::FRONT:
                        agent.position.top += pixelChangeRun, agent.position.bottom += pixelChangeRun;
                        agent.leftEyePosition.top += pixelChangeRun, agent.leftEyePosition.bottom += pixelChangeRun;
                        agent.rightEyePosition.top += pixelChangeRun, agent.rightEyePosition.bottom += pixelChangeRun;
                        break;
                    case Orientation::LEFT:
                        agent.position.left += pixelChangeRun, agent.position.right += pixelChangeRun;
                        agent.leftEyePosition.left += pixelChangeRun, agent.leftEyePosition.right += pixelChangeRun;
                        agent.rightEyePosition.left += pixelChangeRun, agent.rightEyePosition.right += pixelChangeRun;
                        break;
                    case Orientation::RIGHT:
                        agent.position.left -= pixelChangeRun, agent.position.right -= pixelChangeRun;
                        agent.leftEyePosition.left -= pixelChangeRun, agent.leftEyePosition.right -= pixelChangeRun;
                        agent.rightEyePosition.left -= pixelChangeRun, agent.rightEyePosition.right -= pixelChangeRun;
                        break;
                    case Orientation::BACK:
                        agent.position.top -= pixelChangeRun, agent.position.bottom -= pixelChangeRun;
                        agent.leftEyePosition.top -= pixelChangeRun, agent.leftEyePosition.bottom -= pixelChangeRun;
                        agent.rightEyePosition.top -= pixelChangeRun, agent.rightEyePosition.bottom -= pixelChangeRun;
                        break;
                    default:
                        MessageBox(nullptr, "Unknown orientation", "Error", MB_ICONERROR);
                        break;
                    }
                    break;
                case Action::TURN_LEFT:
                    switch (orientation)
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
                    switch (orientation)
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
                case Action::TURN_AROUND:
                    switch (orientation)
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
                case Action::STATIC:
                    // std::this_thread::sleep_for(std::chrono::seconds(2));
                    break;
                case Action::SLEEP:
                    // std::this_thread::sleep_for(std::chrono::seconds(10));
                    break;
                default:
                    MessageBox(nullptr, "Unknown action", "Error", MB_ICONERROR);
                    break;
                }

                agent.has_collided_with_agent2 = false;
                agent.has_collided_with_food = false;
                agent.has_collided_with_water = false;
                agent.has_collided_with_wall = false;
                agent.has_collided_with_predator = false;

                // IDEA: I think whenever he collieded with something, he needs to go through CNN process to determine
                // what he exactly collided with like in real life. For example, he will be presented with some images
                // or could be some videos. Perhaps in the future, he's gonna learn how to drive, and there I could
                // definetely utilize CNN.
                ResolveRectanglesCollision(agent, agent2, Entity::AGENT2, client_width, client_height);
                ResolveRectanglesCollision(agent, food, Entity::FOOD, client_width, client_height);
                ResolveRectanglesCollision(agent, water, Entity::WATER, client_width, client_height);
                ResolveRectanglesCollision(agent, predator, Entity::PREDATOR, client_width, client_height);
                ResolveBoundaryCollision(agent, client_width, client_height);

                // SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(&agent));

                if (agent.has_collided_with_food)
                    PlaySound(TEXT("assets\\eating_sound_effect.wav"), NULL, SND_FILENAME);
                if (agent.has_collided_with_water)
                    PlaySound(TEXT("assets\\gulp-37759.wav"), NULL, SND_FILENAME);

                ++iteration;
                env.Render(i, iteration, action, q_learning.exploration_rate, direction, agent);
                auto [next_state, reward, temp_done] = env.Step(action, agent);
                done = temp_done;

                q_learning.UpdateQtable(state, action, reward, next_state, done);

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

    return 0;
}