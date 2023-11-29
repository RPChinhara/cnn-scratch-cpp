#pragma once

// NOTE: The dimensions of the client area (the inner content area) of the window is 1904 x 1041, on the other hand the size of the entire window (including borders and title bar) is 1920 x 1080 as it set when creating the Windows application. Retrieve by those codes below.
// RECT windowRect;
// GetWindowRect(hwnd, &windowRect);

// int windowWidth = windowRect.right - windowRect.left;
// int windowHeight = windowRect.bottom - windowRect.top;

// RECT clientRect;
// GetClientRect(hwnd, &clientRect);

// int clientWidth = clientRect.right - clientRect.left;
// int clientHeight = clientRect.bottom - clientRect.top;

RECT agent  = { 13, 981, 63, 1031 }; // Left, Top, Right, Bottom coordinates
RECT agent2 = { 1849, 986, 1899, 1036 };
RECT food   = { 5, 5, 55, 55 };
RECT water  = { 1849, 5, 1899, 55 };
RECT bed    = { 5, 976, 71, 1036 };