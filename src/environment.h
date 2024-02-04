#pragma once

#include "action.h"
#include "entity.h"
#include "state.h"

#include <set>
#include <string>
#include <windows.h>

class Environment
{
  public:
    Environment(const LONG client_width, const LONG client_height);
    void Render(const size_t episode, const size_t iteration, Action action, float exploration_rate,
                Direction direction);
    size_t Reset();
    std::tuple<size_t, float, bool> Step(Action action);

    LONG client_width, client_height;
    LONG minLeft = 0;
    LONG maxLeft = client_width - agent_width;
    LONG minTop = 0;
    LONG maxTop = client_height - agent_height;

    size_t numLeftStates = (maxLeft - minLeft) + 1;
    size_t numTopStates = (maxTop - minTop) + 1;

    size_t numThirstStates = 5;
    size_t numHungerStates = 5;
    size_t numEnergyStates = 5;
    size_t numEmotionStates = 3;
    size_t numPhysicalHealthStates = 4;

    size_t numStates = numPhysicalHealthStates * numEmotionStates * numEnergyStates * numHungerStates *
                       numThirstStates * numTopStates * numLeftStates;
    size_t numActions = 7;

  private:
    size_t FlattenState(LONG left, LONG top, ThirstState thirstState, HungerState hungerState, EnergyState energyState,
                        EmotionState emotionState);
    void CalculateReward(const Action action);
    bool CheckTermination();

    bool prevHasCollidedWithWater;
    bool prevHasCollidedWithFood;

    size_t numWaterCollision;
    size_t numFoodCollision;
    size_t numFriendCollision;
    size_t numFriendCollisionWhileHappy;
    size_t numWallCollision;

    size_t numWalk;
    size_t numRun;
    size_t numTurnLeft;
    size_t numTurnRight;
    size_t numTurnAround;
    size_t numStatic;
    size_t numSleep;

    size_t currentState;

    ThirstState thirstState;
    HungerState hungerState;
    EnergyState energyState;
    EmotionState emotionState;
    PhysicalHealthState physicalHealthState;

    std::string thirstStateStr;
    std::string hungerStateStr;
    std::string energyStateStr;
    std::string emotionStateStr;
    std::string physicalHealthStateStr;

    std::string actionStr;

    float reward;

    size_t secondsLived;
    size_t minutesLived;
    size_t hoursLived;
    size_t daysLived;
    // TODO: Do I need Lived?
    size_t secondsLivedWithoutDrinking;
    size_t minutesLivedWithoutDrinking;
    size_t hoursLivedWithoutDrinking;
    size_t daysLivedWithoutDrinking;

    size_t secondsLivedWithoutEating;
    size_t minutesLivedWithoutEating;
    size_t hoursLivedWithoutEating;
    size_t daysLivedWithoutEating;

    size_t secondsLivedWithoutSocializing;
    size_t minutesLivedWithoutSocializing;
    size_t hoursLivedWithoutSocializing;
    size_t daysLivedWithoutSocializing;

    size_t maxDays = 50;
    size_t maxDaysWithoutEating = 43;
    size_t maxDaysWithoutDrinking = 3;

    bool energyLevelBelow3;

    std::set<LONG> seenLefts;
    std::set<LONG> seenTops;
    bool newLeft;
    bool newTop;
};