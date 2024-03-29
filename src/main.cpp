#include "datasets\englishspanish.h"
#include "models\transformer.h"

#include <iostream>
#include <windows.h>

#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")

LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    AllocConsole();

    FILE *file;
    freopen_s(&file, "CONOUT$", "w", stdout);

    SetConsoleOutputCP(CP_UTF8);

    EnglishSpanish englishSpanish = LoadEnglishSpanish();

    for (int i = 0; i < englishSpanish.targetRaw.size(); ++i)
        std::cout << englishSpanish.targetRaw[i] << " " << englishSpanish.contextRaw[i] << std::endl;

    Transformer transformer = Transformer();

    const char CLASS_NAME[] = "WindowClass";
    const char WINDOW_NAME[] = "Dora";

    WNDCLASS wc = {};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    if (!RegisterClass(&wc))
    {
        MessageBox(nullptr, "Window Registration Failed!", "Error!", MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }

    HWND hwnd = CreateWindow(CLASS_NAME, WINDOW_NAME, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 800, 600,
                             nullptr, nullptr, hInstance, nullptr);

    if (hwnd == nullptr)
    {
        MessageBox(nullptr, "Window Creation Failed!", "Error", MB_ICONERROR);
    }
    else
    {
        ShowWindow(hwnd, nCmdShow);
        UpdateWindow(hwnd);
    }

    MSG msg = {};
    while (true)
    {

        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);

            if (msg.message == WM_QUIT)
                return static_cast<int>(msg.wParam);
        }
    }

    fclose(file);
    FreeConsole();

    return static_cast<int>(msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)
    {
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    case WM_PAINT: {

        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        RECT client_rect;
        GetClientRect(hwnd, &client_rect);
        HBRUSH whiteBrush = CreateSolidBrush(RGB(255, 255, 255));
        FillRect(hdc, &client_rect, whiteBrush);
        DeleteObject(whiteBrush);

        // TextOut(hdc, 10, 10, "Hello, Windows!", 15);

        EndPaint(hwnd, &ps);
        break;
    }
    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
    return 0;
}