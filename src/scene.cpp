#include "scene.h"

bool scene::load(renderer* r) {
    floor = mesh(floor_vertices, std::size(floor_vertices), floor_indices, std::size(floor_indices));
    if (!floor.init(r)) {
        logger::log("Failed to init the floor");
        return false;
    }

    agent = mesh(cube_vertices, std::size(cube_vertices), cube_indices, std::size(cube_indices));
    if (!agent.init(r)) {
        logger::log("Failed to init the agent");
        return false;
    }

    return true;
}

void scene::draw(renderer& r, const camera& cam) {
    r.begin_frame({floor, agent}, cam);
    r.end_frame();
}