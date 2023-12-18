#pragma once

#include <windows.h>

void CheckBoundaryCollision(RECT& rect, const int client_width, const int client_height);
void CheckRectanglesCollision(RECT& rect1, const RECT& rect2);