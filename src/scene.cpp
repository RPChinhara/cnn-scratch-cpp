#include "scene.h"

bool scene::load(renderer* r) {
    floor = mesh(floor_vertices, std::size(floor_vertices), floor_indices, std::size(floor_indices));
    if (!floor.init(r))
        return false;

    agent = mesh(agent_vertices, std::size(agent_vertices), agent_indices, std::size(agent_indices));
    if (!agent.init(r))
        return false;

    water = mesh(water_vertices, std::size(water_vertices), water_indices, std::size(water_indices));
    if (!water.init(r))
        return false;

    food = mesh(food_vertices, std::size(food_vertices), food_indices, std::size(food_indices));
    if (!food.init(r))
        return false;

    return true;
}

void scene::draw(renderer& r, const camera& cam) {
    r.begin_frame({floor, agent, water, food}, cam);
    r.end_frame();
}