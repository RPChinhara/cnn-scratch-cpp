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
    Environment(const LONG client_width, const LONG client_height, const Agent &agent);
    void Render(const size_t episode, const size_t iteration, Action action, float exploration_rate,
                const Agent &agent);
    size_t Reset(const Agent &agent);
    std::tuple<size_t, float, bool> Step(Action action, const WinData &winData);

    LONG client_width, client_height;
    LONG minLeft = 0;
    LONG maxLeft;
    LONG minTop = 0;
    LONG maxTop;

    size_t numLeftStates;
    size_t numTopStates;

    size_t numThirstStates = 5;
    size_t numHungerStates = 5;
    size_t numEnergyStates = 5;
    size_t numEmotionStates = 3;

    size_t numStates;
    size_t numActions = 7;

  private:
    size_t FlattenState(LONG left, LONG top, ThirstState thirstState, HungerState hungerState, EnergyState energyState,
                        EmotionState emotionState);
    void CalculateReward(const Action action, const WinData& winData);
    bool CheckTermination(const Agent &agent);

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

    size_t currentState;

    ThirstState thirstState;
    HungerState hungerState;
    EnergyState energyState;
    EmotionState emotionState;

    std::string thirstStateStr;
    std::string hungerStateStr;
    std::string energyStateStr;
    std::string emotionStateStr;

    std::string actionStr;

    float reward;

    size_t secondsLived;
    size_t minutesLived;
    size_t hoursLived;
    size_t daysLived;

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