#include "window.h"
#include "audio_player.h"
#include "entities.h"
#include "environment.h"
#include "physics.h"
#include "q_learning.h"

#include <iostream>
#include <stdexcept>
#include <thread>

// Define a custom message to trigger UI updates
#define WM_UPDATE_DISPLAY (WM_USER + 1)

// Link the necessary libraries
#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "Gdi32.lib")
#pragma comment(lib, "Ole32.lib")
#pragma comment(lib, "User32.lib")

const char Window::CLASS_NAME[] = "EnvWindow";
int Window::window_width  = 1920;
int Window::window_height = 1080;
std::mutex Window::agentMutex; // Initialize the static mutex

Window::Window(HINSTANCE hInst, int nCmdShow) : hInstance(hInst), hwnd(nullptr) {
    // Create a window class
    WNDCLASS wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    if (!RegisterClass(&wc)) {
        MessageBox(nullptr, "Window Registration Failed!", "Error", MB_ICONERROR);
    }

    // Create a window
    hwnd = CreateWindow(
        CLASS_NAME,          // Window class name
        "",                  // Window title
        WS_OVERLAPPEDWINDOW, // Window style
        0,                   // X position
        0,                   // Y position
        window_width,        // Width
        window_height,       // Height
        nullptr,             // Parent window
        nullptr,             // Menu
        hInstance,           // Instance handle
        nullptr              // Additional application data
    );

    if (hwnd == nullptr) {
        MessageBox(nullptr, "Window Creation Failed!", "Error", MB_ICONERROR);
    }

    ShowWindow(hwnd, nCmdShow);
}

int Window::messageLoop() {
    // Main message loop
    AudioPlayer soundPlayer(hwnd);

    if (!soundPlayer.Initialize()) {
        // Handle initialization error
        MessageBox(nullptr, "1", "Error", MB_ICONERROR);
        return -1;
    }

    if (!soundPlayer.LoadAudioData("assets\\mixkit-city-traffic-background-ambience-2930.wav")) {
        // Handle audio data loading error
        MessageBox(nullptr, "2", "Error", MB_ICONERROR);
        return -1;
    }
    
    // Play the sound
    if (!soundPlayer.PlaySound()) {
        // Handle sound playback error
        MessageBox(nullptr, "Failed to play sound!", "Error", MB_ICONERROR);
        return -1;
    }

    // TODO: 実際の時間と合わせる
    // TODO: Recheck if everything is properly implemented by comparing with numpy ver

    // NOTE: Consider creating Window::rl_thread(), and make a variable std::thread rlThread(&Window::rl_thread, this);. Also, I might need to carefully manage shared resources and synchronization to avoid potential issues such as data races.
    std::thread rl_thread([this]() {
        // Reinforcement learning (Q-learining)
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

                // TODO: Maybe I don't need it after all?
                // Lock the mutex before modifying the agent rectangle
                // std::lock_guard<std::mutex> lock(agentMutex);

                // TODO: I think I could use enum for the action?
                // Change agent's position according to the action
                if (action == 2) {
                    agent.top -= 5;  // Move the agent 10 pixels to the top
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

                // Check for collision with the static rectangle
                // if (IsColliding(agent, agent2)) {
                //     ResolveCollision(agent, agent2); // Resolve the collision by adjusting the position of the moving rectangle
                // } else if (IsColliding(agent, water)) {
                //     ResolveCollision(agent, water);
                // } else {
                //     agent.left += 5; // Move the agent 10 pixels to the right
                //     agent.right += 5;
                // }

                // Agent takes the selected action and observes the environment
                auto [next_state, reward, temp_done] = env.step(env.actions[action]);
                done = temp_done;

                // Agent updates the Q-table
                q_learning.update_q_table(state, action, reward, next_state);
                std::cout << q_learning.q_table << std::endl << std::endl;

                env.render();

                total_reward += reward;
                state = next_state;

                // Communicate with the main thread to update the display if needed
                PostMessage(hwnd, WM_UPDATE_DISPLAY, 0, 0);

                // Sleep or yield to give the main thread a chance to process the message
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
                // Handle quitting the application
                // rl_thread.join(); // Wait for the RL thread to finish before exiting
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
            // Lock the mutex before reading the agent rectangle
            // std::lock_guard<std::mutex> lock(agentMutex);

            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);

            // Clear the entire client area with a background color (e.g., white)
            RECT clientRect;
            GetClientRect(hwnd, &clientRect);
            HBRUSH greenBrush = CreateSolidBrush(RGB(34, 139, 34));
            FillRect(hdc, &clientRect, greenBrush);
            DeleteObject(greenBrush);  // Release the brush

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
            // Lock the mutex before reading the agent rectangle
            // std::lock_guard<std::mutex> lock(agentMutex);

            // Redraw the window or perform other UI updates
            InvalidateRect(hwnd, nullptr, TRUE);
            return 0;
        }
        default:
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}