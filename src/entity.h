#pragma once

#include <windows.h>

static RECT agent;
static RECT agent_2;
static RECT bed;
static RECT food;
static RECT water;

static bool has_collided_with_agent_2 = false;
static bool has_collided_with_food = false;
static bool has_collided_with_water = false;