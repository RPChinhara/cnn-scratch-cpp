#pragma once

#include <cstdint>

struct vertex {
    float x, y, z;
};

const vertex floor_vertices[] = {
    { -500.0f, 0.0f, -500.0f },
    { -500.0f, 0.0f,  500.0f },
    {  500.0f, 0.0f, -500.0f },
    {  500.0f, 0.0f,  500.0f }
};

const uint32_t floor_indices[] = {
    0, 1, 2, 2, 1, 3
};

const vertex agent_vertices[] = {
    { -2.5f,  0.0f, -2.5f },
    { -2.5f,  5.0f, -2.5f },
    {  2.5f,  0.0f, -2.5f },
    {  2.5f,  5.0f, -2.5f },
    { -2.5f,  0.0f,  2.5f },
    { -2.5f,  5.0f,  2.5f },
    {  2.5f,  0.0f,  2.5f },
    {  2.5f,  5.0f,  2.5f }
};

const vertex water_vertices[] = {
    { -100.0f, 0.0f, -100.0f },
    { -100.0f, 5.0f, -100.0f },
    { -95.0f, 0.0f, -100.0f },
    { -95.0f, 5.0f, -100.0f },
    { -100.0f, 0.0f, -95.0f },
    { -100.0f, 5.0f, -95.0f },
    { -95.0f, 0.0f, -95.0f },
    { -95.0f, 5.0f, -95.0f }
};

const vertex food_vertices[] = {
    { 95.0f, 0.0f, 95.0f },
    { 95.0f, 5.0f, 95.0f },
    { 100.0f, 0.0f, 95.0f },
    { 100.0f, 5.0f, 95.0f },
    { 95.0f, 0.0f, 100.0f },
    { 95.0f, 5.0f, 100.0f },
    { 100.0f, 0.0f, 100.0f },
    { 100.0f, 5.0f, 100.0f }
};

const uint32_t cube_indices[] = {
    0, 1, 2,  1, 3, 2, // Front face
    4, 6, 5,  5, 6, 7, // Back face
    0, 2, 4,  4, 2, 6, // Bottom face
    1, 5, 3,  3, 5, 7, // Top face
    0, 4, 1,  1, 4, 5, // Left face
    2, 3, 6,  6, 3, 7  // Right face
};