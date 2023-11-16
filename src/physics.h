#pragma once

#include <windows.h>

void CheckBoundaryCollision(RECT& rect, int window_width, int window_height);
bool IsColliding(const RECT& rect1, const RECT& rect2);
void ResolveCollision(RECT& movingRect, const RECT& staticRect);