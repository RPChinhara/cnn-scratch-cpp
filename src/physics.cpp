#include "physics.h"
#include "entity.h"

#include <algorithm>
#include <iostream>

void ResolveBoundaryCollision(RECT& rect, const int client_width, const int client_height)
{
    LONG agent_width = 50, agent_height = 50;
    LONG eyeWidth = 5, LONG eyeHeight = 13;

    // agent           = { 13, (client_height - 13) - agent_height, 13 + agent_width, client_height - 13 };
    // agent_left_eye  = { 53 - agent_eye_width, (client_height - 13) - agent_height + 10, 53, (client_height - 13) - agent_height + 10 + agent_eye_height };
    // agent_right_eye = { 23, (client_height - 13) - agent_height + 10, 23 + agent_eye_width, (client_height - 13) - agent_height + 10 + agent_eye_height };

    if (rect.left < 0) {
        has_collided_with_wall= true;
        
        rect.left = 0;
        rect.right = agent_width;

        // agent_left_eye.left = client_height - agent_height + 10;
        // agent_left_eye.right = client_height - agent_height + 10 + eyeHeight;

        // agent_right_eye.left = client_height - agent_height + 10;
        // agent_right_eye.right = client_height - agent_height + 10 + eyeHeight;
    }
    if (rect.top < 0) {
        has_collided_with_wall = true;

        rect.top = 0;
        rect.bottom = agent_height;
    }
    if (rect.right > client_width) {
        has_collided_with_wall = true;

        rect.left = client_width - agent_width;
        rect.right = client_width;

        // agent_left_eye.left = client_height - agent_height + 10;
        // agent_left_eye.right = client_height - agent_height + 10 + eyeHeight;

        // agent_right_eye.left = client_height - agent_height + 10;
        // agent_right_eye.right = client_height - agent_height + 10 + eyeHeight;
    }
    if (rect.bottom > client_height) {
        has_collided_with_wall = true;

        rect.top = client_height - agent_height;
        rect.bottom = client_height;

        agent_left_eye.top = client_height - agent_height + 10;
        agent_left_eye.bottom = client_height - agent_height + 10 + eyeHeight;

        agent_right_eye.top = client_height - agent_height + 10;
        agent_right_eye.bottom = client_height - agent_height + 10 + eyeHeight;
    }
}

void ResolveRectanglesCollision(RECT& rect1, const RECT& rect2, Entity entity)
{
    if ((rect1.left < rect2.right) && (rect1.right > rect2.left) && (rect1.top < rect2.bottom) && (rect1.bottom > rect2.top)) {

        switch (entity) {
            case AGENT2:
                has_collided_with_agent_2 = true;
                break;
            case FOOD:
                has_collided_with_food = true;
                break;
            case WATER:
                has_collided_with_water = true;
                break;
            default:
                MessageBox(nullptr, "Unknown entity", "Error", MB_ICONERROR);
                break;
        }
        
        int horizontalOverlap = std::min(rect1.right, rect2.right) - std::max(rect1.left, rect2.left);
        int verticalOverlap = std::min(rect1.bottom, rect2.bottom) - std::max(rect1.top, rect2.top);

        if (horizontalOverlap < verticalOverlap) {
            if (rect1.left < rect2.left) {
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