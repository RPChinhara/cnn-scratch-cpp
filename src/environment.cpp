#include "environment.h"

#include <iostream>

void Environment::render() {
    // std::cout << "Current hunger: " << current_state["hunger"] << std::endl;
    // std::cout << "Current thirstiness: " << current_state["thirstiness"] << std::endl;
    // std::cout << "Current mental health: " << current_state["mental health"] << std::endl;
    // std::cout << "Current days lived: " << days_lived << std::endl;
    // std::cout << "Current thirsty days: " << thirsty_days << std::endl;
}

int Environment::reset() {
    // Reset the environment to its initial state
    days_lived          = 0;
    days_without_eating = 0;
    current_state       = 1;
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

//     return std::make_tuple();
// }

    // int Environment::calculate_reward() {
    //     // // Define rewards and penalties based on the environment's state
    //     // if (thirsty_days > max_thirsty_days)
    //     //     return -1; // Penalize for being very thirsty for too long
    //     // else if (days_lived >= max_days)
    //     //     return 1; // Reward for living the desired number of days
    //     // else
    //     //     return 0; // No additional reward or penalty
    // }

    // bool Environment::check_termination() {
    //     // Check if the termination conditions are met
    //     // return days_lived >= max_days || thirsty_days > max_thirsty_days;
    // }

int Environment::map_state(int hunger, int thirstiness, int mental_health) {
    return hunger * 25 + thirstiness * 5 + mental_health;
}

int Environment::update_thirstiness(int action) {
    // Implement how thirstiness changes based on agent's actions
    if (action == 0)
        return 2;
    else if (action == 1)
        return 1;
    else if (action == 2)
        return 3;
    else
        return 5;
}