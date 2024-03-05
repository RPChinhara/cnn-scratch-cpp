#include "physics.h"
#include "entity.h"

#include <algorithm>
#include <iostream>

void ResolveBoundaryCollision(Agent &agent, const LONG client_width, const LONG client_height)
{
    if (agent.position.left < 0)
    {
        std::cout << 1 << " ResolveBoundaryCollision" << '\n';
        std::cout << agent.position.left << '\n';
        agent.has_collided_with_wall = true;

        agent.position.left = 0;
        agent.position.right = agent.width;

        agent.leftEyePosition.left = (agent.position.right - agent.toEyeWidth) - agent.eye_width;
        agent.leftEyePosition.right = agent.position.right - agent.toEyeWidth;

        agent.rightEyePosition.left = agent.toEyeWidth;
        agent.rightEyePosition.right = agent.toEyeWidth + agent.eye_width;
    }

    if (agent.position.top < 0)
    {
        std::cout << 2 << " ResolveBoundaryCollision" << '\n';
        std::cout << agent.position.top << '\n';

        agent.has_collided_with_wall = true;

        agent.position.top = 0;
        agent.position.bottom = agent.height;

        agent.leftEyePosition.top = agent.toEyeHeight;
        agent.leftEyePosition.bottom = agent.toEyeHeight + agent.eye_height;

        agent.rightEyePosition.top = agent.toEyeHeight;
        agent.rightEyePosition.bottom = agent.toEyeHeight + agent.eye_height;
    }

    if (agent.position.right > client_width)
    {
        std::cout << 3 << " ResolveBoundaryCollision" << '\n';
        std::cout << agent.position.right << '\n';

        agent.has_collided_with_wall = true;

        agent.position.left = client_width - agent.width;
        agent.position.right = client_width;

        agent.leftEyePosition.left = (agent.position.right - agent.toEyeWidth) - agent.eye_width;
        agent.leftEyePosition.right = agent.position.right - agent.toEyeWidth;

        agent.rightEyePosition.left = agent.position.left + agent.toEyeWidth;
        agent.rightEyePosition.right = (agent.position.left + agent.toEyeWidth) + agent.eye_width;
    }

    if (agent.position.bottom > client_height)
    {
        std::cout << 4 << " ResolveBoundaryCollision" << '\n';
        std::cout << agent.position.bottom << '\n';

        agent.has_collided_with_wall = true;

        agent.position.top = client_height - agent.height;
        agent.position.bottom = client_height;

        agent.leftEyePosition.top = agent.position.top + agent.toEyeHeight;
        agent.leftEyePosition.bottom = agent.position.top + agent.toEyeHeight + agent.eye_height;

        agent.rightEyePosition.top = agent.position.top + agent.toEyeHeight;
        agent.rightEyePosition.bottom = agent.position.top + agent.toEyeHeight + agent.eye_height;
    }
}

void ResolveRectanglesCollision(Agent &agent, const RECT &entity, Entity entityType, const LONG client_width,
                                const LONG client_height)
{
    if ((agent.position.left < entity.right) && (agent.position.right > entity.left) &&
        (agent.position.top < entity.bottom) && (agent.position.bottom > entity.top))
    {

        switch (entityType)
        {
        case AGENT2:
            agent.has_collided_with_agent2 = true;
            break;
        case FOOD:
            agent.has_collided_with_food = true;
            break;
        case WATER:
            agent.has_collided_with_water = true;
            break;
        default:
            MessageBox(nullptr, "Unknown entity", "Error", MB_ICONERROR);
            break;
        }

        int horizontalOverlap = std::min(agent.position.right, entity.right) - std::max(agent.position.left, entity.left);
        int verticalOverlap = std::min(agent.position.bottom, entity.bottom) - std::max(agent.position.top, entity.top);

        if (horizontalOverlap < verticalOverlap)
        {
            if (agent.position.left < entity.left && agent.previousPosition.left < entity.left)
            {
                std::cout << agent.position.left << " " << agent.position.top << " " << agent.position.right << " "
                          << agent.position.bottom << '\n';
                std::cout << 0.1 << " from Physics" << '\n';
                std::cout << horizontalOverlap << '\n';

                agent.position.left -= horizontalOverlap;
                agent.position.right -= horizontalOverlap;
            }
            else if (agent.position.left < entity.left && agent.previousPosition.left > entity.left)
            {
                std::cout << agent.position.left << " " << agent.position.top << " " << agent.position.right << " "
                          << agent.position.bottom << '\n';
                std::cout << 0.2 << " from Physics" << '\n';
                std::cout << horizontalOverlap << '\n';

                agent.position.left += horizontalOverlap;
                agent.position.right += horizontalOverlap;
            }
            else if (agent.position.left > entity.left && agent.previousPosition.left > entity.left)
            {
                std::cout << agent.position.left << " " << agent.position.top << " " << agent.position.right << " "
                          << agent.position.bottom << '\n';
                std::cout << 0.3 << " from Physics" << '\n';
                std::cout << horizontalOverlap << '\n';

                agent.position.left += horizontalOverlap;
                agent.position.right += horizontalOverlap;
            }
            else if (agent.position.left > entity.left && agent.previousPosition.left < entity.left)
            {
                std::cout << agent.position.left << " " << agent.position.top << " " << agent.position.right << " "
                          << agent.position.bottom << '\n';
                std::cout << 0.4 << " from Physics" << '\n';
                std::cout << horizontalOverlap << '\n';

                agent.position.left -= horizontalOverlap;
                agent.position.right -= horizontalOverlap;
            }
            else if (agent.position.left == entity.right || agent.position.right == entity.left)
            {
                std::cout << " do nothing " << '\n';
            }

            if (agent.position.left < 0)
            {
                std::cout << 1 << " from Physics" << '\n';
                agent.position.left = agent.previousPosition.left;
                agent.position.right = agent.previousPosition.right;

                agent.leftEyePosition.left = (agent.position.right - agent.toEyeWidth) - agent.eye_width;
                agent.leftEyePosition.right = agent.position.right - agent.toEyeWidth;

                agent.rightEyePosition.left = agent.position.left + agent.toEyeWidth;
                agent.rightEyePosition.right = (agent.position.left + agent.toEyeWidth) + agent.eye_width;
            }
            else if (agent.position.right > client_width)
            {
                std::cout << 2 << " from Physics" << '\n';

                agent.position.left = agent.previousPosition.left;
                agent.position.right = agent.previousPosition.right;

                agent.leftEyePosition.left = (agent.position.right - agent.toEyeWidth) - agent.eye_width;
                agent.leftEyePosition.right = agent.position.right - agent.toEyeWidth;

                agent.rightEyePosition.left = agent.position.left + agent.toEyeWidth;
                agent.rightEyePosition.right = (agent.position.left + agent.toEyeWidth) + agent.eye_width;
            }
            else
            {
                agent.leftEyePosition.left = (agent.position.right - agent.toEyeWidth) - agent.eye_width;
                agent.leftEyePosition.right = agent.position.right - agent.toEyeWidth;

                agent.rightEyePosition.left = agent.position.left + agent.toEyeWidth;
                agent.rightEyePosition.right = (agent.position.left + agent.toEyeWidth) + agent.eye_width;
            }
        }
        else
        {
            if (agent.position.top < entity.top && agent.previousPosition.top < entity.top)
            {
                std::cout << agent.position.left << " " << agent.position.top << " " << agent.position.right << " "
                          << agent.position.bottom << '\n';
                std::cout << 0.5 << " from Physics" << '\n';
                std::cout << verticalOverlap << '\n';

                agent.position.top -= verticalOverlap;
                agent.position.bottom -= verticalOverlap;
            }
            else if (agent.position.top < entity.top && agent.previousPosition.top > entity.top)
            {
                std::cout << agent.position.left << " " << agent.position.top << " " << agent.position.right << " "
                          << agent.position.bottom << '\n';
                std::cout << 0.6 << " from Physics" << '\n';
                std::cout << verticalOverlap << '\n';

                agent.position.top += verticalOverlap;
                agent.position.bottom += verticalOverlap;
            }
            else if (agent.position.top > entity.top && agent.previousPosition.top > entity.top)
            {
                std::cout << agent.position.left << " " << agent.position.top << " " << agent.position.right << " "
                          << agent.position.bottom << '\n';
                std::cout << 0.7 << " from Physics" << '\n';
                std::cout << verticalOverlap << '\n';

                agent.position.top += verticalOverlap;
                agent.position.bottom += verticalOverlap;
            }
            else if (agent.position.top > entity.top && agent.previousPosition.top > entity.top)
            {
                std::cout << agent.position.left << " " << agent.position.top << " " << agent.position.right << " "
                          << agent.position.bottom << '\n';
                std::cout << 0.8 << " from Physics" << '\n';
                std::cout << verticalOverlap << '\n';

                agent.position.top -= verticalOverlap;
                agent.position.bottom -= verticalOverlap;
            }
            else if (agent.position.top == entity.bottom || agent.position.bottom == entity.top)
            {
                std::cout << " do nothing " << '\n';
            }

            if (agent.position.top < 0)
            {
                std::cout << 3 << " from Physics" << '\n';

                agent.position.top = agent.previousPosition.top;
                agent.position.bottom = agent.previousPosition.bottom;

                agent.leftEyePosition.top = agent.position.top + agent.toEyeHeight;
                agent.leftEyePosition.bottom = agent.position.top + agent.toEyeHeight + agent.eye_height;

                agent.rightEyePosition.top = agent.position.top + agent.toEyeHeight;
                agent.rightEyePosition.bottom = agent.position.top + agent.toEyeHeight + agent.eye_height;
            }
            else if (agent.position.bottom > client_height)
            {
                std::cout << 4 << " from Physics" << '\n';

                agent.position.top = agent.previousPosition.top;
                agent.position.bottom = agent.previousPosition.bottom;

                agent.leftEyePosition.top = agent.position.top + agent.toEyeHeight;
                agent.leftEyePosition.bottom = agent.position.top + agent.toEyeHeight + agent.eye_height;

                agent.rightEyePosition.top = agent.position.top + agent.toEyeHeight;
                agent.rightEyePosition.bottom = agent.position.top + agent.toEyeHeight + agent.eye_height;
            }
            else
            {
                agent.leftEyePosition.top = agent.position.top + agent.toEyeHeight;
                agent.leftEyePosition.bottom = agent.position.top + agent.toEyeHeight + agent.eye_height;

                agent.rightEyePosition.top = agent.position.top + agent.toEyeHeight;
                agent.rightEyePosition.bottom = agent.position.top + agent.toEyeHeight + agent.eye_height;
            }
        }
    }
}