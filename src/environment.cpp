#include "environment.h"

#include <iostream>
#include <windows.h>

void Environment::render() {
    std::cout << "Days Lived: " << days_lived << std::endl;
    std::cout << "Current State: " << states[current_state] << std::endl;
    std::cout << "Days Without Eating: " << days_without_eating << " days" << std::endl << std::endl;
}

int Environment::reset() {
    // Reset the environment to its initial state
    days_lived          = 0;
    days_without_eating = 0;
    current_state       = std::distance(states.begin(), std::find(states.begin(), states.end(), "neutral"));
    return current_state;
}

std::tuple<int, int, bool> Environment::step(const std::string& action) {
    // Use std::find to search for the action in the actions
    auto it = std::find(actions.begin(), actions.end(), action);

    // If action is not in actions
    if (it == actions.end()) {
        MessageBox(NULL, "Invalid action.", "Error", MB_ICONERROR);
        ExitProcess(1);
    }
    
    // TODO: if the action was "eat", and current state was "full" it should be penalized
    // Implement how the state changes based on the agent's actions
    if (action == "eat" && current_state != std::distance(states.begin(), std::find(states.begin(), states.end(), "full"))) // TODO: simplify this code using lambda?
        current_state = std::min(current_state + 1, num_states - 1);
    else if (action == "do_nothing" && current_state != std::distance(states.begin(), std::find(states.begin(), states.end(), "hungry")))
        current_state = std::max(current_state - 1, 0);

    // Increment days lived by 1
    days_lived += 1;

    // Check if the agent has eaten
    if (action == "eat")
        days_without_eating = 0;
    else
        days_without_eating += 1;

    // TODO: I could implement update_thirstiness() which implements how thirstiness changes based on agent's actions e.g.,
    // if (action == 0)
    //     return 2;
    // else if (action == 1)
    //     return 1;
    // else if (action == 2)
    //     return 3;
    // else
    //     return 5;

    // Calculate the reward based on the environment's state
    int reward = calculate_reward();
    bool done  = check_termination();

    return std::make_tuple(current_state, reward, done);
}

int Environment::calculate_reward() {
    // Define rewards and penalties based on the environment's state
    if (current_state == std::distance(states.begin(), std::find(states.begin(), states.end(), "hungry")))
        return -1; // Penalize for being hungry
    else if (days_without_eating >= max_days_without_eating)
        return -1;
    else if (days_lived >= max_days)
        return 1; // Reward for living the desired number of days
    else
        return 0; // No additional reward or penalty
}

bool Environment::check_termination() {
    // Check if the termination conditions are met
    return days_lived >= max_days || days_without_eating >= max_days_without_eating;
}