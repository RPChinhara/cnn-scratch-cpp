#pragma once

#include <windows.h>

class Agent;
class Entity;

void ResolveBoundaryCollision(Agent &agent, const LONG clientWidth, const LONG clientHeight);
void ResolveRectanglesCollision(Agent &agent, const Entity &entity, const LONG clientWidth, const LONG clientHeight);