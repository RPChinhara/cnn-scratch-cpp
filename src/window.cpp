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

Window::Window(HINSTANCE hInst, int nCmdShow) : hInstance(hInst), hwnd(nullptr) {
    // Create a window class
    WNDCLASS wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    if (!RegisterClass(&wc)) {
        MessageBox(nullptr, "Window Registration Failed!", "Error", MB_ICONERROR);
    }

    // TODO: How to hide the taskbar when window is displayed?

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
    
    // TODO: No sound playing
    // Play the sound
    if (!soundPlayer.PlaySound()) {
        // Handle sound playback error
        MessageBox(nullptr, "Failed to play sound!", "Error", MB_ICONERROR);
        return -1;
    }

    // TODO: I need to introduce sounds in order so that it resembles real world, e.g., sounds of possible predators so that it can scare him.
    // TODO: He needs to make sounds like cats do.
    // TODO: 実際の時間と合わせる
    // TODO: Recheck if everything is properly implemented by comparing with numpy ver
    // TODO: I need to set the camera focused on him, and spawn objects e.g., food, water, possible friends, and predators, etc to let him explore the environment.

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
                rl_thread.join(); // Wait for the RL thread to finish before exiting
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
        case WM_KEYDOWN: {
            // TODO: He may need to classify the objects before take actions e.g., if it's food he'd eat.
            // TODO: How to implement five senses specially touch, smell, taste as these could influence fudamental actions like eating the right food.
            // Check which key was pressed

            // TODO: Possibly foward I he needs to run in that case I need to set pixes for example 5 pixels for walk, and 10 pixels for run.
            int key = wParam;
            if (key == VK_RIGHT) { // Move right when the right arrow key is pressed
                
                // Check for collision with the static rectangle
                if (IsColliding(agent, agent2)) {
                    ResolveCollision(agent, agent2); // Resolve the collision by adjusting the position of the moving rectangle
                } else if (IsColliding(agent, water)) {
                    ResolveCollision(agent, water);
                } else {
                    agent.left += 5; // Move the agent 10 pixels to the right
                    agent.right += 5;
                }

                 // Check for boundary collision
                CheckBoundaryCollision(agent, window_width, window_height);

                InvalidateRect(hwnd, nullptr, TRUE); // Redraw the updated rectangle
            }
            if (key == VK_LEFT) {
                agent.left -= 5;
                agent.right -= 5;
                InvalidateRect(hwnd, nullptr, TRUE);
            }
            if (key == VK_UP) {
                agent.top -= 5;
                agent.bottom -= 5;
                InvalidateRect(hwnd, nullptr, TRUE);
            }
            if (key == VK_DOWN) {
                agent.top += 5;
                agent.bottom += 5;
                InvalidateRect(hwnd, nullptr, TRUE);
            }
            return 0;
        }
        case WM_PAINT: {
            // TODO: Use Direct2D next, and Direct3D 9 or Direct3D 10 for 3D?
            // TODO: Draw days lived, current_state, days_without_eating, and location on the simulation screen? or create menu?
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            
            // Clear the entire client area with a background color (e.g., white)
            RECT clientRect;
            GetClientRect(hwnd, &clientRect);
            FillRect(hdc, &clientRect, CreateSolidBrush(RGB(34, 139, 34)));

            // Draw a rectangle
            FillRect(hdc, &bed, CreateSolidBrush(RGB(255, 255, 255)));
            FillRect(hdc, &agent, CreateSolidBrush(RGB(218, 171, 145)));
            FillRect(hdc, &agent2, CreateSolidBrush(RGB(218, 171, 145)));
            FillRect(hdc, &food, CreateSolidBrush(RGB(255, 0, 0)));
            FillRect(hdc, &water, CreateSolidBrush(RGB(0, 0, 255)));

            EndPaint(hwnd, &ps);
            return 0;
        }
        case WM_SIZE: {
            // RECT clientRect;
            // GetClientRect(hwnd, &clientRect);
            // // Handle window resizing
            // int clientWidth = LOWORD(lParam);
            // int clientHeight = HIWORD(lParam);

            // // Calculate the position of the agent as a percentage of the client area
            // agentXPercent = 0.005; // 0.5% from the left
            // agentYPercent = 0.995; // 99.5% from the top

            // // Calculate the new positions and sizes of your rectangles here
            // agentWidth = static_cast<int>(clientWidth * 50 / clientWidth);
            // agentHeight = static_cast<int>(clientHeight * 50 / clientHeight);

            // agent2 = {
            //     static_cast<int>(clientRect.right * agentXPercent),
            //     static_cast<int>(clientRect.bottom * agentYPercent) - agentHeight,
            //     static_cast<int>(clientRect.right * agentXPercent) + agentWidth,
            //     static_cast<int>(clientRect.bottom * agentYPercent)
            // };

            // Redraw the scene
            InvalidateRect(hwnd, nullptr, TRUE);
            return 0;
        }
        case WM_UPDATE_DISPLAY:
            // Redraw the window or perform other UI updates
            InvalidateRect(hwnd, nullptr, TRUE);
            return 0;
        default:
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}