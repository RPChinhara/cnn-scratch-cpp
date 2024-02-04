#pragma once

#include <windows.h>

enum Entity
{
    AGENT,
    AGENT2,
    BED,
    FOOD,
    WATER,
    PREDATOR
};

enum Orientation
{
    FRONT,
    LEFT,
    RIGHT,
    BACK
};

enum Direction
{
    NORTH,
    SOUTH,
    EAST,
    WEST
};

// TODO: I think global variables are messed up. Learn how to handle.
inline RECT agent;
inline RECT agent_previous;
inline RECT agent_left_eye;
inline RECT agent_right_eye;
inline RECT agent2;
inline RECT bed;
inline RECT food;
inline RECT water;
inline RECT predator;

inline LONG agent_width = 50, agent_height = 50;
inline LONG agent_eye_width = 5, agent_eye_height = 13;
inline LONG bed_width = 66, bed_height = 60;
inline LONG food_width = 50, food_height = 50;
inline LONG water_width = 50, water_height = 50;
inline LONG predator_width = 60, predator_height = 60;

inline LONG agentToEyeWidth = 13;
inline LONG agentToEyeHeight = 10;
inline LONG borderToAgent = 13;
inline LONG borderToEntities = 5;

// IDEA: Maybe make class Entity so that I can avoid these below global variables?
inline bool has_collided_with_agent2 = false;
inline bool has_collided_with_food = false;
inline bool has_collided_with_water = false;
inline bool has_collided_with_wall = false;
inline bool has_collided_with_predator = false;

inline bool render_agent_left_eye = true;
inline bool render_agent_right_eye = true;