#pragma once

#include <windows.h>

class Agent
{
  public:
    Agent() = default;

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

    bool has_collided_with_agent2 = false;
    bool has_collided_with_food = false;
    bool has_collided_with_water = false;
    bool has_collided_with_wall = false;
    bool has_collided_with_predator = false;

    bool render_agent_left_eye = true;
    bool render_agent_right_eye = true;
};