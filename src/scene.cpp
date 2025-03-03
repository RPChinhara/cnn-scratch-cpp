#include "scene.h"

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