#include "environment.h"

#include <iostream>
#include <windows.h>

// TODO: Learning about why animals and human's primary purposes for living I guess making good environment might be key? If they need to adapt to environment, it needs to change over time e.g., global warming, we have to adapt to new techs.
// TODO: Try open-ended RL?
// TODO: Use PowerPlay? https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00313/full
// TODO: Try automatically inventing or discovering problems in a way inspired by the playful behavior of animals and humans (PowerPlay).
// TODO: Run environments the GPU instead of CPU for like multi-agent reinforcement learning (MARL) research.
// TODO: The book 'Why Greatness Cannot Be Planned' proposes to follow the interesting and the novel instead. The authors developed an algorithm called novelty search, where instead of optimizing an objective, the agent just tries out behaviors that are as novel to it as possible, but I realized that paper 'Reward is Enough' came 6 years after this so not sure...
// TODO: Maybe there might be RL within Rl? e.g., within the states he decide to do puzzle game in order to learn new words at this point use another RL? I mean human is also like that meaning it’s like layers of RL. In order to live, you have to work, and within the work you have to do certatin tasks.
// TODO: When actions is set talk, set states as current situation such as holding a leaf sets alphabet as actions so that it can chose a word 'lead!' like baby would say.

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
        MessageBox(nullptr, "Invalid action.", "Error", MB_ICONERROR);
        ExitProcess(1);
    }
    
    // TODO: An agent should address new goals forever.
    // TODO: He needs to take various nutrition such as carbohydrates, proteins, fats, vitamins, and minerals through food these adequate nutrition is crucial for energy production, growth, and the maintenance and repair of bodily tissues so perhaps in the future I have to program so that not he only eats, but have to eat in good balance.

    // TODO: if the action was "eat", and current state was "full" it should be penalized
    // Implement how the state changes based on the agent's actions
    if (action == "eat" && current_state != std::distance(states.begin(), std::find(states.begin(), states.end(), "full"))) // TODO: simplify this code using lambda?
        current_state = std::min(current_state + 1, num_states - 1);
    else if (action != "eat" && current_state != std::distance(states.begin(), std::find(states.begin(), states.end(), "hungry")))
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
    if (current_state == std::distance(states.begin(), std::find(states.begin(), states.end(), "hungry")) || days_without_eating >= max_days_without_eating)
        return -1; // Penalize for being hungry
    else if (days_lived >= max_days) {
        days_lived = 0;
        return 1; // Reward for living the desired number of days
    } else
        return 0; // No additional reward or penalty
}

bool Environment::check_termination() {
    // Check if the termination conditions are met
    return days_without_eating >= max_days_without_eating;
}