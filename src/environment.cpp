#include "environment.h"

#include <iostream>
#include <windows.h>

void Environment::Render()
{
    std::cout << "Days Lived: " << daysLived << std::endl;
    std::cout << "Current State: " << states[currentState] << std::endl;
    std::cout << "Days Without Eating: " << daysWithoutEating << " days" << std::endl << std::endl;
}

int Environment::Reset()
{
    daysLived          = 0;
    daysWithoutEating = 0;
    currentState       = std::distance(states.begin(), std::find(states.begin(), states.end(), "neutral"));
    return currentState;
}

std::tuple<int, int, bool> Environment::Step(const std::string& action)
{
    auto it = std::find(actions.begin(), actions.end(), action);

    if (it == actions.end()) {
        MessageBox(nullptr, "Invalid action.", "Error", MB_ICONERROR);
        ExitProcess(1);
    }
    
    if (action == "eat" && currentState != std::distance(states.begin(), std::find(states.begin(), states.end(), "full")))
        currentState = std::min(currentState + 1, numStates - 1);
    else if (action != "eat" && currentState != std::distance(states.begin(), std::find(states.begin(), states.end(), "hungry")))
        currentState = std::max(currentState - 1, 0);

    daysLived += 1;

    if (action == "eat")
        daysWithoutEating = 0;
    else
        daysWithoutEating += 1;

    int reward = CalculateReward();
    bool done = CheckTermination();

    return std::make_tuple(currentState, reward, done);
}

int Environment::CalculateReward()
{
    if (currentState == std::distance(states.begin(), std::find(states.begin(), states.end(), "hungry")) || daysWithoutEating >= maxDaysWithoutEating) {
        return -1;
    } else if (daysLived >= maxDays) {
        daysLived = 0;
        return 1;
    } else {
        return 0;
    }
}

bool Environment::CheckTermination()
{
    return daysWithoutEating >= maxDaysWithoutEating;
}