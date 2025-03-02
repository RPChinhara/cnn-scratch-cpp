#pragma once

#include <cstdint>

struct vertex {
    float x, y, z;
};

const vertex floor_vertices[] = {
    { -5.0f, 0.0f, -5.0f },
    { -5.0f, 0.0f,  5.0f },
    {  5.0f, 0.0f, -5.0f },
    {  5.0f, 0.0f,  5.0f }
};

const uint32_t floor_indices[] = {
    0, 1, 2, 2, 1, 3
};

const vertex cube_vertices[] = {
    { -2.5f, -2.5f, -2.5f },
    { -2.5f,  2.5f, -2.5f },
    {  2.5f, -2.5f, -2.5f },
    {  2.5f,  2.5f, -2.5f },
    { -2.5f, -2.5f,  2.5f },
    { -2.5f,  2.5f,  2.5f },
    {  2.5f, -2.5f,  2.5f },
    {  2.5f,  2.5f,  2.5f }
};

const uint32_t cube_indices[] = {
    0, 1, 2,  1, 3, 2,
    4, 6, 5,  5, 6, 7,
    0, 2, 4,  4, 2, 6,
    1, 5, 3,  3, 5, 7,
    0, 4, 1,  1, 4, 5,
    2, 3, 6,  6, 3, 7
};