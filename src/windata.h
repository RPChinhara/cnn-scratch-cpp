#pragma once

#include "entities\agent.h"
#include "entities\bed.h"
#include "entities\building.h"
#include "entities\entity.h"
#include "entities\food.h"
#include "entities\mod.h"
#include "entities\street.h"
#include "entities\water.h"

struct WinData
{
    WinData() = default;
    Agent agent;
    Bed bed;
    Building building;
    Food food;
    Mod mod;
    Street street;
    Water water;
};