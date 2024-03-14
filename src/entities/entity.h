#pragma once

#include <windows.h>

class Entity
{
  public:
    enum Type
    {
        AGENT2,
        BED,
        FOOD,
        WATER,
    };

    RECT position;
    Type type;
};