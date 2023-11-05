#include "environment.h"

#include <iostream>

void Environment::render() {
    std::cout << "Days Lived: " << days_lived << std::endl;
    std::cout << "Current State: " << current_state << std::endl;
    std::cout << "Days Without Eating: " << days_without_eating << std::endl;
}

int Environment::reset() {
    // Reset the environment to its initial state
    days_lived          = 0;
    days_without_eating = 0;
    current_state       = std::distance(states.begin(), std::find(states.begin(), states.end(), "neutral"));
    return 0;
}

// std::tuple<States, int, bool> Environment::step(int action) {
//     // // Reduce thirstiness based on action level and update other states
//     // current_state["thirstiness"] = update_thirstiness(action);
//     // days_lived += 1;

//     // // Check if the agent has been very thirsty for too long
//     // if (current_state["thirstiness"] == 2)
//     //     thirsty_days += 1;
//     // else
//     //     thirsty_days = 0;
    
//     // // Define rewards and penalties
//     // int reward = calculate_reward();
//     // bool done  = check_termination();

//     // if (done) {
//     //     MessageBox(NULL, "The game is over. Please reset the environment.", "Error", MB_ICONERROR | MB_OK);
//     //     ExitProcess(1);
//     // }

    //TODO: I could implement update_thirstiness() which implements how thirstiness changes based on agent's actions e.g.,
    // if (action == 0)
    //     return 2;
    // else if (action == 1)
    //     return 1;
    // else if (action == 2)
    //     return 3;
    // else
    //     return 5;

//     return std::make_tuple();
// }

int Environment::calculate_reward() {
    // Define rewards and penalties based on the environment's state
    if (current_state == std::distance(states.begin(), std::find(states.begin(), states.end(), "hungry")))
        return -1; // Penalize for being hungry
    else if (days_lived >= max_days)
        return 1; // Reward for living the desired number of days
    else
        return 0; // No additional reward or penalty
}

bool Environment::check_termination() {
    // Check if the termination conditions are met
    return days_lived >= max_days;
}