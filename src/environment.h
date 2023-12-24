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

    size_t numActions = 4;
    size_t numStates = 3;
    
private:
    int CalculateReward();
    bool CheckTermination();

    Action actions;
    State states;
    size_t currentState = State::NEUTRAL;
    std::string currentStateStr;
    std::string currentAction;
    int reward;
    size_t daysLived = 0;
    size_t daysWithoutEating = 0;
    size_t daysWithoutDrinking = 0;
    size_t maxDays = 50;
    size_t maxDaysWithoutEating = 43;
    size_t maxDaysWithoutDrinking = 3;
};