#pragma once

#include <windows.h>

enum Entity {
    AGENT,
    AGENT2,
    BED,
    FOOD,
    WATER
};

enum Orientation {
    FRONT,
    LEFT,
    RIGHT,
    BACK,
};

inline RECT agent;
inline RECT agent_eye_1;
inline RECT agent_eye_2;
inline RECT agent_2;
inline RECT bed;
inline RECT food;
inline RECT water;

inline LONG agent_width = 50, agent_height = 50;
inline LONG agent_eye_width = 5, agent_eye_height = 13;
inline LONG bed_width = 66, bed_height = 60;
inline LONG food_width = 50, food_height = 50;
inline LONG water_width = 50, water_height = 50;

inline bool has_collided_with_agent_2 = false;
inline bool has_collided_with_food = false;
inline bool has_collided_with_water = false;
inline bool has_collided_with_wall = false;