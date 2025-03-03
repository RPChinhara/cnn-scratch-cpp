#pragma once

#include "mesh.h"
#include "renderer.h"
#include "logger.h"

class scene {
public:
    bool load(renderer* r);
    void draw(renderer& r, const camera& cam);

private:
    mesh floor;
    mesh agent;
    mesh water;
    mesh food;
    mesh predator;
};