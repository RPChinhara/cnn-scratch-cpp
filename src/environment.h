#pragma once

#include "action.h"
#include "entity.h"

#include <set>
#include <string>
#include <windows.h>

class Environment
{
public:
    Environment(const LONG client_width, const LONG client_height);
    void Render(const size_t iteration, Action action, float exploration_rate, Direction direction);
    size_t Reset();
    std::tuple<size_t, float, bool> Step(Action action);

    LONG client_width, client_height;
    LONG minLeft = 0;
    LONG maxLeft = client_width - agent_width;
    LONG minTop = 0;
    LONG maxTop = client_height - agent_height;

    size_t numThirstStates = 3;
    size_t numHungerStates = 3;
    size_t numEnergyStates = 3;
    size_t numLeftStates = (maxLeft - minLeft) + 1;
    size_t numTopStates = (maxTop - minTop) + 1;

    size_t numStates = numThirstStates * numHungerStates * numEnergyStates * numLeftStates * numTopStates;
    size_t numActions = 6;
    
private:
    size_t FlattenState(size_t hungerState, size_t thirstState, size_t energyState, LONG left,LONG top);
    void CalculateReward();
    bool CheckTermination();

    size_t numWaterCollision;
    size_t numFoodCollision;
    size_t numFriendCollision;
    size_t numWallCollision;
    size_t numMoveForward;
    size_t numTurnLeft;
    size_t numTurnRight;
    size_t numTurnAround;
    size_t numStatic;
    size_t currentState;
    size_t thirstState;
    size_t hungerState;
    size_t energyState;
    std::string thirstStateStr;
    std::string hungerStateStr;
    std::string energyStateStr;
    std::string actionStr;
    float reward;
    size_t daysLived;
    size_t daysWithoutEating;
    size_t daysWithoutDrinking;
    size_t maxDays = 50;
    size_t maxDaysWithoutEating = 43;
    size_t maxDaysWithoutDrinking = 3;

    std::set<LONG> seenLefts;
    std::set<LONG> seenTops;
    bool newLeft;
    bool newTop;
};