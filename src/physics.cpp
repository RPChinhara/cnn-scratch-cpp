#include "physics.h"

#include <algorithm>

void CheckBoundaryCollision(RECT& rect, int window_width, int window_height) {
    // Check and handle collisions with the window boundaries
    if (rect.left < 0) {
        rect.left = 0;
        // rect.right = rect.left + (rect.right - rect.left);
        rect.right = 50;
    }
    if (rect.top < 0) {
        rect.top = 0;
        // rect.bottom = rect.top + (rect.bottom - rect.top);
        rect.bottom = 50;
    }
    if (rect.right > 1904) {
        rect.right = 1904;
        // rect.left = rect.right - (rect.right - rect.left);
        rect.left = 1854;
    }
    if (rect.bottom > 1041) {
        rect.bottom = 1041;
        // rect.top = rect.bottom - (rect.bottom - rect.top);
        rect.top = 991;
    }
}

bool IsColliding(const RECT& rect1, const RECT& rect2) {
    // Check for collision between two rectangles
    return (rect1.left < rect2.right &&
            rect1.right > rect2.left &&
            rect1.top < rect2.bottom &&
            rect1.bottom > rect2.top);
}

void ResolveCollision(RECT& movingRect, const RECT& staticRect) {
    // Determine the horizontal and vertical distances between rectangles
    int horizontalDistance = std::min(abs(staticRect.right - movingRect.left), abs(movingRect.right - staticRect.left));
    int verticalDistance = std::min(abs(staticRect.bottom - movingRect.top), abs(movingRect.bottom - staticRect.top));

    if (horizontalDistance < 10 || verticalDistance < 10) {
        // If the rectangles are too close, stop the moving rectangle
        return;
    }

    // Move the rectangle to the right
    movingRect.left += 10;
    movingRect.right += 10;
}