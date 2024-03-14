#pragma once

#include "entity.h"

#include <windows.h>

class Bed : public Entity
{
  public:
    const LONG width = 66;
    const LONG height = 60;

    Bed() = default;
    Bed(LONG clientHeight, LONG borderToEntities)
    {
        position = {borderToEntities, (clientHeight - borderToEntities) - height, borderToEntities + width,
                    clientHeight - borderToEntities};
        type = BED;
    }
    Bed &operator=(const Bed &other)
    {
        if (this != &other)
        {
            position = other.position;
            type = other.type;
        }
        return *this;
    }
};