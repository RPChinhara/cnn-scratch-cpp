#pragma once

#include <windows.h>

inline RECT agent;
inline RECT agent_2;
inline RECT bed;
inline RECT food;
inline RECT water;

inline bool has_collided_with_agent_2 = false;
inline bool has_collided_with_food = false;
inline bool has_collided_with_water = false;