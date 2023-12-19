#pragma once

#include <windows.h>

void ResolveBoundaryCollision(RECT& rect, const int client_width, const int client_height);
void ResolveRectanglesCollision(RECT& rect1, const RECT& rect2);