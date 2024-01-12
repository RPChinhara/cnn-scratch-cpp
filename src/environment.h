#pragma once

#include "entity.h"

#include <string>
#include <windows.h>

enum State {
    HUNGRY,
    NEUTRAL,
    FULL
};

enum ThirstState {
    THIRSTY,
    QUENCHED,
    HYDRATED
};

// enum HungerState {
//     HUNGRY,
//     NEUTRAL,
//     FULL
// };

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
    Environment(const LONG client_width, const LONG client_height);
    void Render();
    size_t Reset();
    std::tuple<size_t, int, bool> Step(const size_t action);

    LONG client_width, client_height;
    LONG minLeft = 0;
    LONG maxLeft = client_width - agent_width;
    LONG minTop = 0;
    LONG maxTop = client_height - agent_height;

    size_t numHungerStates = 3;
    size_t numThirstStates = 3;
    size_t numLeftLevels = maxLeft - minLeft;
    size_t numTopLevels = maxTop - minTop;

    size_t numStates = numHungerStates * numThirstStates * numLeftLevels * numTopLevels;
    size_t numActions = 5;
    
private:
    size_t Environment::FlattenState(size_t hungerState, size_t thirstState, LONG left,LONG top);
    int CalculateReward();
    bool CheckTermination();

    size_t hungerState = State::NEUTRAL;
    size_t thirstState = ThirstState::QUENCHED;
    size_t currentState = FlattenState(hungerState, thirstState, agent.left, agent.top);
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