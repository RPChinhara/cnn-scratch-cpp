#pragma once

#include "entity.h"

#include <windows.h>

class Food : public Entity
{
  public:
    const LONG width = 50;
    const LONG height = 50;

    Food() = default;
    Food(LONG borderToEntities)
    {
        position = {borderToEntities, borderToEntities, borderToEntities + width, borderToEntities + height};
        type = FOOD;
    }
    Food &operator=(const Food &other)
    {
        if (this != &other)
        {
            position = other.position;
            type = other.type;
        }
        return *this;
    }
};