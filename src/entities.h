#pragma once

#include "entities\agent.h"
#include "entities\agent2.h"
#include "entities\bed.h"
#include "entities\building.h"
#include "entities\entity.h"
#include "entities\food.h"
#include "entities\water.h"

struct Entities
{
    Entities() = default;
    Agent agent;
    Agent2 agent2;
    Bed bed;
    Building building;
    Food food;
    Water water;
};