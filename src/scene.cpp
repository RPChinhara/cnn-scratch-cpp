#include "scene.h"

const vertex floor_vertices[] = {
    { -500.0f, 0.0f, -500.0f },
    { -500.0f, 0.0f,  500.0f },
    {  500.0f, 0.0f, -500.0f },
    {  500.0f, 0.0f,  500.0f }
};

const uint32_t rect_indices[] = {
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

const vertex predator_vertices[] = {
    { 45.0f, 0.0f, 45.0f },
    { 45.0f, 5.0f, 45.0f },
    { 50.0f, 0.0f, 45.0f },
    { 50.0f, 5.0f, 45.0f },
    { 45.0f, 0.0f, 50.0f },
    { 45.0f, 5.0f, 50.0f },
    { 50.0f, 0.0f, 50.0f },
    { 50.0f, 5.0f, 50.0f }
};

const uint32_t cube_indices[] = {
    0, 1, 2,  1, 3, 2, // Front face
    4, 6, 5,  5, 6, 7, // Back face
    0, 2, 4,  4, 2, 6, // Bottom face
    1, 5, 3,  3, 5, 7, // Top face
    0, 4, 1,  1, 4, 5, // Left face
    2, 3, 6,  6, 3, 7  // Right face
};

bool scene::load(renderer* r) {
    floor = mesh(floor_vertices, std::size(floor_vertices), rect_indices, std::size(rect_indices));
    if (!floor.init(r))
        return false;

    agent = mesh(agent_vertices, std::size(agent_vertices), cube_indices, std::size(cube_indices));
    if (!agent.init(r))
        return false;

    water = mesh(water_vertices, std::size(water_vertices), cube_indices, std::size(cube_indices));
    if (!water.init(r))
        return false;

    food = mesh(food_vertices, std::size(food_vertices), cube_indices, std::size(cube_indices));
    if (!food.init(r))
        return false;

    predator = mesh(predator_vertices, std::size(predator_vertices), cube_indices, std::size(cube_indices));
    if (!predator.init(r))
        return false;

    return true;
}

void scene::draw(renderer& r, const camera& cam) {
    r.begin_frame({ floor, agent, water, food, predator }, cam);
    r.end_frame();
}