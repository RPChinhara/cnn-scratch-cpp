#pragma once

#include <windows.h>

enum Direction
{
    NORTH,
    SOUTH,
    EAST,
    WEST
};

enum Orientation
{
    FRONT,
    LEFT,
    RIGHT,
    BACK
};

class Agent
{
  public:
    Agent() = default;
    Agent(LONG clientWidth, LONG clientHeight)
    {
        LONG halfWidth = width / 2;
        LONG halfHeight = height / 2;
        LONG halfClientWidth = clientWidth / 2;
        LONG halfClientHeight = clientHeight / 2;

        position = {halfClientWidth - halfWidth, halfClientHeight - halfHeight, halfClientWidth - halfWidth + width,
                    halfClientHeight - halfHeight + height};

        leftEyePosition = {(position.right - toEyeWidth) - eye_width, halfClientHeight - halfHeight + toEyeHeight,
                           position.right - toEyeWidth, halfClientHeight - halfHeight + toEyeHeight + eye_height};

        rightEyePosition = {position.left + toEyeWidth, halfClientHeight - halfHeight + toEyeHeight,
                            (position.left + toEyeWidth) + eye_width,
                            halfClientHeight - halfHeight + toEyeHeight + eye_height};
    }
    Agent &operator=(const Agent &other)
    {
        if (this != &other)
        {
            direction = other.direction;
            orientation = other.orientation;
            position = other.position;
            previousPosition = other.previousPosition;
            leftEyePosition = other.leftEyePosition;
            rightEyePosition = other.rightEyePosition;

            has_collided_with_food = other.has_collided_with_food;
            has_collided_with_mod = other.has_collided_with_mod;
            has_collided_with_wall = other.has_collided_with_wall;
            has_collided_with_water = other.has_collided_with_water;

            render_agent_left_eye = other.render_agent_left_eye;
            render_agent_right_eye = other.render_agent_right_eye;
        }
        return *this;
    }

    Direction direction = Direction::SOUTH;
    Orientation orientation = Orientation::FRONT;
    RECT position;
    RECT previousPosition;
    RECT leftEyePosition;
    RECT rightEyePosition;

    const LONG width = 50;
    const LONG height = 50;

    const LONG eye_width = 5;
    const LONG eye_height = 13;

    const LONG toEyeWidth = 13;
    const LONG toEyeHeight = 10;

    bool has_collided_with_food = false;
    bool has_collided_with_mod = false;
    bool has_collided_with_wall = false;
    bool has_collided_with_water = false;

    bool render_agent_left_eye = true;
    bool render_agent_right_eye = true;
};