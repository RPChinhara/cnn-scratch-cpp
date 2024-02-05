#pragma once

#include "agent.h"
#include "entity.h"

void ResolveBoundaryCollision(Agent &agent, const LONG client_width, const LONG client_height);
void ResolveRectanglesCollision(Agent &agent, const RECT &rect2, Entity entity, const LONG client_width,
                                const LONG client_height);