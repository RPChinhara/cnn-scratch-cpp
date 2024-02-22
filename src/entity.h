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

inline RECT agent2;
inline RECT bed;
inline RECT food;
inline RECT water;
inline RECT predator;

inline LONG bed_width = 66, bed_height = 60;
inline LONG food_width = 50, food_height = 50;
inline LONG water_width = 50, water_height = 50;
inline LONG predator_width = 60, predator_height = 60;

inline LONG borderToAgent = 13;
inline LONG borderToEntities = 5;