#include "physics.h"

#include <iostream>

void CheckBoundaryCollision(RECT& rect, const int client_width, const int client_height)
{
    LONG agent_width = 50, agent_height = 50;

    if (rect.left < 0) {
        rect.left = 0;
        rect.right = agent_width;
    }
    if (rect.top < 0) {
        rect.top = 0;
        rect.bottom = agent_height;
    }
    if (rect.right > client_width) {
        rect.right = client_width;
        rect.left = client_width - agent_width;
    }
    if (rect.bottom > client_height) {
        rect.bottom = client_height;
        rect.top = client_height - agent_height;
    }
}

void CheckRectanglesCollision(RECT& rect1, const RECT& rect2)
{
    if ((rect1.left < rect2.right) && (rect1.right > rect2.left) &&
        (rect1.top < rect2.bottom) && (rect1.bottom > rect2.top)) {
        
        int horizontalOverlap = std::min(rect1.right, rect2.right) - std::max(rect1.left, rect2.left);
        int verticalOverlap = std::min(rect1.bottom, rect2.bottom) - std::max(rect1.top, rect2.top);

        if (horizontalOverlap < verticalOverlap) {
            if (rect1.left < rect2.left) {
                std::cout << "horizontalOverlap: " << horizontalOverlap << std::endl;
                rect1.left -= horizontalOverlap;
                rect1.right -= horizontalOverlap;
            } else {
                rect1.left += horizontalOverlap;
                rect1.right += horizontalOverlap;
            }
        } else {
            if (rect1.top < rect2.top) {
                rect1.top -= verticalOverlap;
                rect1.bottom -= verticalOverlap;
            } else {
                rect1.top += verticalOverlap;
                rect1.bottom += verticalOverlap;
            }
        }
    }
}