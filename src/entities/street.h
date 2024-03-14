#pragma once

#include "entity.h"

#include <windows.h>

class Street : public Entity
{
  public:
    const LONG width = 3000;
    const LONG height = 120;

    Street() = default;
    Street(LONG clientWidth, LONG clientHeight)
    {
        position = {0, clientHeight / 2, width, clientHeight / 2 + height};
    }
    Street &operator=(const Street &other)
    {
        if (this != &other)
        {
            position = other.position;
        }
        return *this;
    }
};