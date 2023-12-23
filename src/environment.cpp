#include "environment.h"
#include "entity.h"

#include <iostream>
#include <windows.h>

void Environment::Render()
{
    if (current_state == State::HUNGRY)
        current_state_str = "hungry";
    else if (current_state == State::NEUTRAL)
        current_state_str = "neutral";
    else if (current_state == State::FULL)
        current_state_str = "full";

    std::cout << "Current State:         " << current_state_str << std::endl;
    std::cout << "Current Action:        " << current_action << std::endl;
    std::cout << "Days Lived:            " << days_lived << " days" << std::endl;
    std::cout << "Days Without Drinking: " << days_without_drinking << " days" << std::endl;
    std::cout << "Days Without Eating:   " << days_without_eating << " days" << std::endl << std::endl;
}

size_t Environment::Reset()
{
    days_lived = 0;
    days_without_eating = 0;
    current_state = State::NEUTRAL;
    return current_state;
}

std::tuple<size_t, int, bool> Environment::Step(const size_t action)
{
    if (action == Action::MOVE_UP)
        current_action = "move_up";
    else if (action == Action::MOVE_DOWN)
        current_action = "move_down";
    else if (action == Action::MOVE_LEFT)
        current_action = "move_left";
    else if (action == Action::MOVE_LEFT)
        current_action = "move_right";
    
    Render();

    if (has_collided_with_food && current_state != State::FULL)
        current_state = std::min(current_state + 1, num_states - 1);
    else if (!has_collided_with_food && current_state != State::HUNGRY)
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
    if (current_state == State::HUNGRY && days_without_eating >= 3) {
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