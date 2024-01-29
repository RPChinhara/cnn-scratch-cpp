#pragma once

void ResolveBoundaryCollision(RECT& rect, const LONG client_width, const LONG client_height);
void ResolveRectanglesCollision(RECT& rect1, const RECT& rect2, Entity entity, const LONG client_width, const LONG client_height);