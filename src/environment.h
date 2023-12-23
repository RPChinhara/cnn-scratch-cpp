#pragma once

#include <string>

enum Action {
    MOVE_UP,
    MOVE_DOWN,
    MOVE_LEFT,
    MOVE_RIGHT
};

enum State {
    HUNGRY,
    NEUTRAL,
    FULL,
};

class Environment
{
public:
    void Render();
    size_t Reset();
    std::tuple<size_t, int, bool> Step(const size_t action);

    size_t num_actions = 4;
    size_t num_states = 3;
    
private:
    int CalculateReward();
    bool CheckTermination();

    Action actions;
    State states;
    size_t current_state = State::NEUTRAL;
    std::string current_state_str;
    std::string current_action;
    size_t days_lived = 0;
    size_t max_days = 50;
    size_t days_without_eating = 0;
    size_t days_without_drinking = 0;
    size_t max_days_without_eating = 43;
    size_t max_days_without_drinking = 3;
};