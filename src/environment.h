#pragma once

#include <string>

enum State {
    HUNGRY,
    NEUTRAL,
    FULL
};

enum Action {
    MOVE_FORWARD,
    TURN_LEFT,
    TURN_RIGHT,
    TURN_AROUND,
    STATIC
};

class Environment
{
public:
    void Render();
    size_t Reset();
    std::tuple<size_t, int, bool> Step(const size_t action);

    size_t numStates = 3;
    size_t numActions = 5;

    size_t num_hunger_levels = 3;
    size_t num_thirst_levels = 3;
    size_t num_coordinates = 4;
    size_t num_actions = 5;
    
private:
    int CalculateReward();
    bool CheckTermination();

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