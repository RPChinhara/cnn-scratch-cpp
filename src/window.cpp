#include "window.h"
#include "audio_player.h"
#include "entities.h"
#include "environment.h"
#include "physics.h"
#include "q_learning.h"

#include <iostream>
#include <stdexcept>
#include <thread>

#define WM_UPDATE_DISPLAY (WM_USER + 1)

#pragma comment(lib, "Gdi32.lib")
#pragma comment(lib, "User32.lib")

const char Window::CLASS_NAME[] = "EnvWindow";
int Window::window_width  = 1920;
int Window::window_height = 1080;
std::mutex Window::agentMutex;

Window::Window(HINSTANCE hInst, int nCmdShow) : hInstance(hInst), hwnd(nullptr) {
    WNDCLASS wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    if (!RegisterClass(&wc)) {
        MessageBox(nullptr, "Window Registration Failed!", "Error", MB_ICONERROR);
    }

    hwnd = CreateWindow(
        CLASS_NAME,
        "",
        WS_OVERLAPPEDWINDOW,
        0,
        0,
        window_width,
        window_height,
        nullptr,
        nullptr,
        hInstance,
        nullptr
    );

    if (hwnd == nullptr) {
        MessageBox(nullptr, "Window Creation Failed!", "Error", MB_ICONERROR);
    }

    ShowWindow(hwnd, nCmdShow);
}

int Window::messageLoop() {
    AudioPlayer soundPlayer(hwnd);

    if (!soundPlayer.Initialize()) {
        MessageBox(nullptr, "1", "Error", MB_ICONERROR);
        return -1;
    }

    if (!soundPlayer.LoadAudioData("assets\\mixkit-city-traffic-background-ambience-2930.wav")) {
        MessageBox(nullptr, "2", "Error", MB_ICONERROR);
        return -1;
    }
    
    if (!soundPlayer.PlaySound()) {
        MessageBox(nullptr, "Failed to play sound!", "Error", MB_ICONERROR);
        return -1;
    }

    std::thread rl_thread([this]() {
        Environment env = Environment();
        QLearning q_learning = QLearning(env.num_states, env.num_actions);

        unsigned int num_episodes = 1000;

        std::cout << "------------------- HEAD -------------------" << std::endl;

        for (int i = 0; i < num_episodes; ++i) {
            auto state = env.reset();
            bool done = false;
            int total_reward = 0;

            while (!done) {
                unsigned int action = q_learning.choose_action(state);
                std::cout << "action: " << action << std::endl;

                // std::lock_guard<std::mutex> lock(agentMutex);

                if (action == 2) {
                    agent.top -= 5;
                    agent.bottom -= 5;
                } else if (action == 3) {
                    agent.top += 5;
                    agent.bottom += 5;
                } else if (action == 4) {
                    agent.left -= 5;
                    agent.right -= 5;
                } else if (action == 5) {
                    agent.left += 5;
                    agent.right += 5;
                }

                CheckBoundaryCollision(agent, window_width, window_height);

                // if (IsColliding(agent, agent2)) {
                //     ResolveCollision(agent, agent2);
                // } else if (IsColliding(agent, water)) {
                //     ResolveCollision(agent, water);
                // } else {
                //     agent.left += 5;
                //     agent.right += 5;
                // }

                auto [next_state, reward, temp_done] = env.step(env.actions[action]);
                done = temp_done;

                q_learning.update_q_table(state, action, reward, next_state);
                std::cout << q_learning.q_table << std::endl << std::endl;

                env.render();

                total_reward += reward;
                state = next_state;

                PostMessage(hwnd, WM_UPDATE_DISPLAY, 0, 0);

                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }

            std::cout << "Episode " << i + 1 << ": Total Reward = " << total_reward << std::endl << std::endl;
        }
    });

    while(true) {
        MSG msg = {};

        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);

            if (msg.message == WM_QUIT) {
                // rl_thread.join();
                return static_cast<int>(msg.wParam);
            }
        }
    }

    return 0;
}

LRESULT CALLBACK Window::WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        case WM_PAINT: {
            // std::lock_guard<std::mutex> lock(agentMutex);

            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);

            RECT clientRect;
            GetClientRect(hwnd, &clientRect);
            HBRUSH greenBrush = CreateSolidBrush(RGB(34, 139, 34));
            FillRect(hdc, &clientRect, greenBrush);
            DeleteObject(greenBrush);

            HBRUSH whiteBrush = CreateSolidBrush(RGB(255, 255, 255));
            FillRect(hdc, &bed, whiteBrush);
            DeleteObject(whiteBrush);

            HBRUSH pinkBrush = CreateSolidBrush(RGB(209, 163, 164));
            FillRect(hdc, &agent, pinkBrush);
            FillRect(hdc, &agent2, pinkBrush);
            DeleteObject(pinkBrush);

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
            // std::lock_guard<std::mutex> lock(agentMutex);

            InvalidateRect(hwnd, nullptr, TRUE);
            return 0;
        }
        default:
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}