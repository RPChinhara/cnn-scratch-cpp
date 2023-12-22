#include "environment.h"
#include "entity.h"

#include <iostream>
#include <windows.h>

void Environment::Render()
{
    std::cout << "Current State:         " << states[current_state] << std::endl;
    std::cout << "Current Action:        " << current_action << std::endl;
    std::cout << "Days Lived:            " << days_lived << " days" << std::endl;
    std::cout << "Days Without Drinking: " << days_without_drinking << " days" << std::endl;
    std::cout << "Days Without Eating:   " << days_without_eating << " days" << std::endl << std::endl;
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
    current_action = action;
    
    Render();

    if (has_collided_with_food && current_state != std::distance(states.begin(), std::find(states.begin(), states.end(), "full")))
        current_state = std::min(current_state + 1, num_states - 1);
    else if (!has_collided_with_food && current_state != std::distance(states.begin(), std::find(states.begin(), states.end(), "hungry")))
        current_state = std::max(current_state - 1, static_cast<size_t>(0));

    days_lived += 1;

    if (has_collided_with_food)
        days_without_eating = 0;
    else
        days_without_eating += 1;

    if (has_collided_with_water)
        days_without_drinking = 0;
    else
        days_without_drinking += 1;

    int reward = CalculateReward();
    bool done = CheckTermination();

    return std::make_tuple(current_state, reward, done);
}

int Environment::CalculateReward()
{
    if (current_state == std::distance(states.begin(), std::find(states.begin(), states.end(), "hungry")) && days_without_eating >= 3) {
        return -1;
    } else if (days_lived >= max_days) {
        days_lived = 0;
        return 1;
    } else if (has_collided_with_agent_2 || has_collided_with_food || has_collided_with_water) {
        return 1;
    } else {
        return 0;
    }
}

bool Environment::CheckTermination()
{
    return days_without_eating >= max_days_without_eating;
}