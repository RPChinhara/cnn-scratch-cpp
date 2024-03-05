#pragma once

#include <windows.h>

enum Entity
{
    AGENT2,
    BED,
    FOOD,
    WATER,
};

// class Agent2
// {
//     RECT position;

//     const LONG width = 50;
//     const LONG height = 50;
//     Entity entity = Entity::AGENT2
// };

// class Bed
// {
//     RECT position;

//     const LONG width = 66;
//     const LONG height = 60;
//     Entity entity = Entity::Bed
// };

// class Food
// {
//     RECT position;

//     const LONG width = 50;
//     const LONG height = 50;
//     Entity entity = Entity::Food
// };

// class Water
// {
//     RECT position;

//     const LONG width = 50;
//     const LONG height = 50;
//     Entity entity = Entity::Water
// };

inline RECT agent2;
inline RECT bed;
inline RECT food;
inline RECT water;

inline LONG bed_width = 66, bed_height = 60;
inline LONG food_width = 50, food_height = 50;
inline LONG water_width = 50, water_height = 50;