#include "environment.h"
#include "entity.h"

#include <iostream>
#include <windows.h>

void Environment::Render()
{
    if (currentState == State::HUNGRY)
        currentStateStr = "hungry";
    else if (currentState == State::NEUTRAL)
        currentStateStr = "neutral";
    else if (currentState == State::FULL)
        currentStateStr = "full";

    std::cout << "Current State:         " << currentStateStr << std::endl;
    std::cout << "Current Action:        " << currentAction << std::endl;
    std::cout << "Reward:                " << reward << std::endl;
    std::cout << "Days Lived:            " << daysLived << " days" << std::endl;
    std::cout << "Days Without Drinking: " << daysWithoutDrinking << " days" << std::endl;
    std::cout << "Days Without Eating:   " << daysWithoutEating << " days" << std::endl << std::endl;
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
        case Action::MOVE_UP:
            currentAction = "move_up";
            break;
        case Action::MOVE_DOWN:
            currentAction = "move_down";
            break;
        case Action::MOVE_LEFT:
            currentAction = "move_left";
            break;
        case Action::MOVE_RIGHT:
            currentAction = "move_right";
            break;
        default:
            MessageBox(nullptr, "Unknown action", "Error", MB_ICONERROR);
            break;
    }

    Render();

    if (has_collided_with_food && currentState != State::FULL)
        currentState = std::min(currentState + 1, numStates - 1);
    else if (!has_collided_with_food && currentState != State::HUNGRY)
        currentState = std::max(currentState - 1, static_cast<size_t>(0));

    reward = CalculateReward();
    bool done = CheckTermination();

    daysLived += 1;

    if (has_collided_with_food)
        daysWithoutEating = 0;
    else
        daysWithoutEating += 1;

    if (has_collided_with_water)
        daysWithoutDrinking = 0;
    else
        daysWithoutDrinking += 1;

    return std::make_tuple(currentState, reward, done);
}

int Environment::CalculateReward()
{
    if (currentState == State::HUNGRY && daysWithoutEating >= 3 || currentState == State::FULL && has_collided_with_food) {
        return -1;
    } else if (daysLived >= maxDays) {
        daysLived = 0;
        return 1;
    } else if (has_collided_with_agent_2 || has_collided_with_food || has_collided_with_water) {
        return 1;
    } else {
        return 0;
    }
}

bool Environment::CheckTermination()
{
    return daysWithoutEating >= maxDaysWithoutEating;
}