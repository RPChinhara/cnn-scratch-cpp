#pragma once

class Building
{
  public:
    int x;
    int y;

    Building() = default;
    Building(int x, int y)
    {
        this->x = x;
        this->y = y;
    }
};