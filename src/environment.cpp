#include "environment.h"

#include <iostream>
#include <windows.h>

Environment::Environment() {
    days_lived                   = 0;
    thirsty_days                 = 0;
    max_days                     = 50;
    max_thirsty_days             = 3;
    current_state["thirstiness"] = "neutral";
}

void Environment::render() {
    std::cout << "Current guess: " << std::endl;
    std::cout << "Attempts made: " << std::endl;
}

std::unordered_map<std::string, std::string> Environment::reset() {
    days_lived                   = 0;
    thirsty_days                 = 0;
    current_state["thirstiness"] = "neutral";

    return current_state;
}

std::tuple<std::string, int, bool> Environment::step(int action) {
    // Reduce thirstiness based on action level and update other states
    current_state["thirstiness"] = update_thirstiness(action);
    days_lived += 1;

    // Check if the agent has been very thirsty for too long
    if (current_state["thirstiness"] == "very thirsty")
        thirsty_days += 1;
    else
        thirsty_days = 0;
    
    // Define rewards and penalties
    int reward = calculate_reward();
    bool done  = check_termination();

    if (done) {
        MessageBox(NULL, "The game is over. Please reset the environment.", "Error", MB_ICONERROR | MB_OK);
        ExitProcess(1);
    }

    return std::make_tuple(current_state["thirstiness"], reward, done);
}

int Environment::calculate_reward() {
    // Define rewards and penalties based on the environment's state
    if (thirsty_days > max_thirsty_days)
        return -1; // Penalize for being very thirsty for too long
    else if (days_lived >= max_days)
        return 1; // Reward for living the desired number of days
    else
        return 0; // No additional reward or penalty
}

bool Environment::check_termination() {
    // Check if the termination conditions are met
    return days_lived >= max_days || thirsty_days > max_thirsty_days;
}

std::string Environment::update_thirstiness(int action) {
    // Implement how thirstiness changes based on agent's actions
    if (action == 0)
        return "neutral";
    else if (action == 1)
        return "thirsty";
    else if (action == 2)
        return "very thirsty";
    else
        return "unknown";
}