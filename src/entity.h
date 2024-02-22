#pragma once

#include <windows.h>

enum Entity
{
    AGENT2,
    BED,
    FOOD,
    WATER,
    PREDATOR
};

// class Agent2
// {
//     RECT position;

//     const LONG width = 50;
//     const LONG height = 50;
// };

// class Bed
// {
//     RECT position;

//     const LONG width = 66;
//     const LONG height = 60;
// };

// class Food
// {
//     RECT position;

//     const LONG width = 50;
//     const LONG height = 50;
// };

// class Water
// {
//     RECT position;

//     const LONG width = 50;
//     const LONG height = 50;
// };

// class Predator
// {
//     RECT position;

//     const LONG width = 60;
//     const LONG height = 60;
// };

inline RECT agent2;
inline RECT bed;
inline RECT food;
inline RECT water;
inline RECT predator;

inline LONG bed_width = 66, bed_height = 60;
inline LONG food_width = 50, food_height = 50;
inline LONG water_width = 50, water_height = 50;
inline LONG predator_width = 60, predator_height = 60;