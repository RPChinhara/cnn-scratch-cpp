#pragma once

#include <mutex>
#include <windows.h>

class Window {
public:
    Window(HINSTANCE hInst, int nCmdShow);
    int messageLoop();
private:
    HWND hwnd;
    HINSTANCE hInstance;
    static const char CLASS_NAME[];
    static int window_width;
    static int window_height;
    static std::mutex agentMutex; // Mutex to synchronize access to agent
    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
};