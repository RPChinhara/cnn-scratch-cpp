#include "environment.h"

#include <iostream>
#include <windows.h>

void Environment::Render()
{
    std::cout << "Days Lived: " << days_lived << std::endl;
    std::cout << "Current State: " << states[current_state] << std::endl;
    std::cout << "Days Without Eating: " << days_without_eating << " days" << std::endl << std::endl;
}

size_t Environment::Reset()
{
    days_lived = 0;
    days_without_eating = 0;
    current_state = std::distance(states.begin(), std::find(states.begin(), states.end(), "neutral"));
    return current_state;
}

std::tuple<size_t, int, bool> Environment::Step(const std::string& action)
{
    auto it = std::find(actions.begin(), actions.end(), action);

    if (it == actions.end()) {
        MessageBox(nullptr, "Invalid action.", "Error", MB_ICONERROR);
        ExitProcess(1);
    }
    
    if (action == "eat" && current_state != std::distance(states.begin(), std::find(states.begin(), states.end(), "full")))
        current_state = std::min(static_cast<size_t>(current_state + 1), static_cast<size_t>(num_states - 1));
    else if (action != "eat" && current_state != std::distance(states.begin(), std::find(states.begin(), states.end(), "hungry")))
        current_state = std::max(static_cast<size_t>(current_state) - 1, static_cast<size_t>(0));

    days_lived += 1;

    if (action == "eat")
        days_without_eating = 0;
    else
        days_without_eating += 1;

    int reward = CalculateReward();
    bool done = CheckTermination();

    return std::make_tuple(current_state, reward, done);
}

int Environment::CalculateReward()
{
    if (current_state == std::distance(states.begin(), std::find(states.begin(), states.end(), "hungry")) || days_without_eating >= max_days_without_eating) {
        return -1;
    } else if (days_lived >= max_days) {
        days_lived = 0;
        return 1;
    } else {
        return 0;
    }
}

bool Environment::CheckTermination()
{
    return days_without_eating >= max_days_without_eating;
}