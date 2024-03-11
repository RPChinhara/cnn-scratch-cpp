#pragma once

#include "entity.h"

void ResolveBoundaryCollision(Agent &agent, const LONG client_width, const LONG client_height);
void ResolveRectanglesCollision(Agent &agent, const Entity &entity, const LONG client_width, const LONG client_height);