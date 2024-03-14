#pragma once

#include "entity.h"

#include <windows.h>

class Water : public Entity
{
  public:
    const LONG width = 50;
    const LONG height = 50;

    Water() = default;
    Water(LONG clientWidth, LONG clientHeight, LONG borderToEntities)
    {
        position = {(clientWidth - borderToEntities) - width, borderToEntities, clientWidth - borderToEntities,
                    borderToEntities + height};
        type = WATER;
    }
    Water &operator=(const Water &other)
    {
        if (this != &other)
        {
            position = other.position;
            type = other.type;
        }
        return *this;
    }
};