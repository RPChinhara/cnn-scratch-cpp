#pragma once

#include "entity.h"

#include <windows.h>

class Mod : public Entity
{
  public:
    const LONG width = 50;
    const LONG height = 50;

    Mod() = default;
    Mod(LONG clientWidth, LONG clientHeight, LONG borderToEntities)
    {
        position = {(clientWidth - borderToEntities) - width, (clientHeight - borderToEntities) - height,
                    clientWidth - borderToEntities, clientHeight - borderToEntities};
        type = MOD;
    }
    Mod &operator=(const Mod &other)
    {
        if (this != &other)
        {
            position = other.position;
            type = other.type;
        }
        return *this;
    }
};
