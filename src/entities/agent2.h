#pragma once

#include "entity.h"

#include <windows.h>

class Agent2 : public Entity
{
  public:
    const LONG width = 50;
    const LONG height = 50;

    Agent2() = default;
    Agent2(LONG clientWidth, LONG clientHeight, LONG borderToEntities)
    {
        position = {(clientWidth - borderToEntities) - width, (clientHeight - borderToEntities) - height,
                    clientWidth - borderToEntities, clientHeight - borderToEntities};
        type = AGENT2;
    }
    Agent2 &operator=(const Agent2 &other)
    {
        if (this != &other)
        {
            position = other.position;
            type = other.type;
        }
        return *this;
    }
};
