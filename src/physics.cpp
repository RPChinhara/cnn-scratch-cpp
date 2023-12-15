#include "physics.h"

void CheckBoundaryCollision(RECT& rect)
{
    LONG client_width = 1904, client_height = 1041;
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