#include "environment.h"
#include "entity.h"

#include <chrono>
#include <iostream>
#include <windows.h>

inline std::chrono::time_point<std::chrono::high_resolution_clock> lifeStartTime;
inline std::chrono::time_point<std::chrono::high_resolution_clock> lifeEndTime;
inline std::chrono::hours::rep hours;

void Environment::Render()
{
    switch (currentState) {
        case State::HUNGRY:
            currentStateStr = "hungry";
            break;
        case State::NEUTRAL:
            currentStateStr = "neutral";
            break;
        case State::FULL:
            currentStateStr = "full";
            break;
        default:
            MessageBox(nullptr, "Unknown state", "Error", MB_ICONERROR);
            break;
    }

    auto lifeEndTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(lifeEndTime - lifeStartTime);

    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration % std::chrono::seconds(1)).count();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count() % 60;
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration).count() % 60;
    hours = std::chrono::duration_cast<std::chrono::hours>(duration).count() % 24;
    auto days = std::chrono::duration_cast<std::chrono::hours>(duration).count() / 24;

    std::cout << "Current State:         " << currentStateStr << std::endl;
    std::cout << "Current Action:        " << currentAction << std::endl;
    std::cout << "Reward:                " << reward << std::endl;
    std::cout << "Days Lived:            " << days << " days, " << hours << " hours, " << minutes << " minutes, " << seconds << " seconds, and " << milliseconds << " milliseconds" << std::endl;
    std::cout << "Days Without Drinking: " << days << " days, " << hours << " hours, " << minutes << " minutes, " << seconds << " seconds, and " << milliseconds << " milliseconds" << std::endl;
    std::cout << "Days Without Eating:   " << days << " days, " << hours << " hours, " << minutes << " minutes, " << seconds << " seconds, and " << milliseconds << " milliseconds" << std::endl;
}

size_t Environment::Reset()
{
    daysLived = 0;
    daysWithoutEating = 0;
    currentState = State::NEUTRAL;
    return currentState;
}

std::tuple<size_t, int, bool> Environment::Step(const size_t action)
{
    switch (action) {
        case Action::MOVE_FORWARD:
            currentAction = "move forward";
            break;
        case Action::TURN_LEFT:
            currentAction = "turn left";
            break;
        case Action::TURN_RIGHT:
            currentAction = "turn right";
            break;
        case Action::TURN_AROUND:
            currentAction = "turn around";
            break;
        case Action::STATIC:
            currentAction = "static";
            break;
        default:
            MessageBox(nullptr, "Unknown action", "Error", MB_ICONERROR);
            break;
    }

    Render();

    if (has_collided_with_food && currentState != State::FULL)
        currentState = std::min(currentState + 1, numStates - 1);
    else if (hours >= 3 && currentState != State::HUNGRY)
        currentState = std::max(currentState - 1, static_cast<size_t>(0));

    reward = CalculateReward();
    bool done = CheckTermination();

    // if (has_collided_with_food)
    //     daysWithoutEating = 0;
    // else
    //     daysWithoutEating += 1;

    // if (has_collided_with_water)
    //     daysWithoutDrinking = 0;
    // else
    //     daysWithoutDrinking += 1;

    return std::make_tuple(currentState, reward, done);
}

int Environment::CalculateReward()
{
    int reward = 0;

    if (daysLived > maxDays)
        reward += 1;
    if (currentState == State::HUNGRY && has_collided_with_food || currentState == State::NEUTRAL && has_collided_with_food)
        reward += 1;
    if (currentState == State::HUNGRY && !has_collided_with_food)
        reward += -1;
    if (currentState == State::HUNGRY && hours >= 3)
        reward += -1;
    if (currentState == State::FULL && has_collided_with_food)
        reward += -1;
    if (has_collided_with_wall)
        reward += -1;
    else
        reward += 0;

    return reward;
}

bool Environment::CheckTermination()
{
    if (daysLived >= maxDays)
        daysLived = 0;
    
    return daysWithoutEating >= maxDaysWithoutEating || daysLived >= maxDays;
}