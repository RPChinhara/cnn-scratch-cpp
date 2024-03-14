#pragma once

#include <windows.h>

class Entity
{
  public:
    enum Type
    {
        BED,
        FOOD,
        MOD,
        WATER,
    };

    RECT position;
    Type type;
};