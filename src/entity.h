#pragma once

#include <windows.h>

enum Direction
{
    NORTH,
    SOUTH,
    EAST,
    WEST
};

enum Orientation
{
    FRONT,
    LEFT,
    RIGHT,
    BACK
};

class Agent
{
  public:
    Agent() = default;
    Agent(LONG clientWidth, LONG clientHeight, LONG borderToAgent)
    {
        position = {borderToAgent, (clientHeight - borderToAgent) - height, borderToAgent + width,
                    clientHeight - borderToAgent};

        leftEyePosition = {(position.right - toEyeWidth) - eye_width,
                           (clientHeight - borderToAgent) - height + toEyeHeight, position.right - toEyeWidth,
                           (clientHeight - borderToAgent) - height + toEyeHeight + eye_height};

        rightEyePosition = {position.left + toEyeWidth, (clientHeight - borderToAgent) - height + toEyeHeight,
                            (position.left + toEyeWidth) + eye_width,
                            (clientHeight - borderToAgent) - height + toEyeHeight + eye_height};
    }
    Agent &operator=(const Agent &other)
    {
        if (this != &other)
        {
            direction = other.direction;
            orientation = other.orientation;
            position = other.position;
            previousPosition = other.previousPosition;
            leftEyePosition = other.leftEyePosition;
            rightEyePosition = other.rightEyePosition;

            has_collided_with_agent2 = other.has_collided_with_agent2;
            has_collided_with_food = other.has_collided_with_food;
            has_collided_with_water = other.has_collided_with_water;
            has_collided_with_wall = other.has_collided_with_wall;

            render_agent_left_eye = other.render_agent_left_eye;
            render_agent_right_eye = other.render_agent_right_eye;
        }
        return *this;
    }

    Direction direction = Direction::SOUTH;
    Orientation orientation = Orientation::FRONT;
    RECT position;
    RECT previousPosition;
    RECT leftEyePosition;
    RECT rightEyePosition;

    const LONG width = 50;
    const LONG height = 50;

    const LONG eye_width = 5;
    const LONG eye_height = 13;

    const LONG toEyeWidth = 13;
    const LONG toEyeHeight = 10;

    bool has_collided_with_agent2 = false;
    bool has_collided_with_food = false;
    bool has_collided_with_water = false;
    bool has_collided_with_wall = false;

    bool render_agent_left_eye = true;
    bool render_agent_right_eye = true;
};

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

struct Entities
{
    Entities() = default;
    Agent agent;
    Agent2 agent2;
    Bed bed;
    Food food;
    Water water;
};