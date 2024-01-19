#include "environment.h"
#include "entity.h"

#include <chrono>
#include <iostream>
#include <windows.h>

inline std::chrono::time_point<std::chrono::high_resolution_clock> lifeStartTime;
inline std::chrono::time_point<std::chrono::high_resolution_clock> lifeEndTime;
inline std::chrono::hours::rep hours;

Environment::Environment(const LONG client_width, const LONG client_height) : client_width(client_width), client_height(client_height) {
}

void Environment::Render(const size_t iteration, const size_t action, float exploration_rate)
{
    switch (action) {
        case Action::MOVE_FORWARD:
            actionStr = "move forward";
            break;
        case Action::TURN_LEFT:
            actionStr = "turn left";
            break;
        case Action::TURN_RIGHT:
            actionStr = "turn right";
            break;
        case Action::TURN_AROUND:
            actionStr = "turn around";
            break;
        case Action::STATIC:
            actionStr = "static";
            break;
        default:
            MessageBox(nullptr, "Unknown action", "Error", MB_ICONERROR);
            break;
    }

    switch (thirstState) {
        case ThirstState::THIRSTY:
            thirstStateStr = "thirsty";
            break;
        case ThirstState::QUENCHED:
            thirstStateStr = "quenched";
            break;
        case ThirstState::HYDRATED:
            thirstStateStr = "hydrated";
            break;
        default:
            MessageBox(nullptr, "Unknown thirst state", "Error", MB_ICONERROR);
            break;
    }

    switch (hungerState) {
        case HungerState::HUNGRY:
            hungerStateStr = "hungry";
            break;
        case HungerState::NEUTRAL:
            hungerStateStr = "neutral";
            break;
        case HungerState::FULL:
            hungerStateStr = "full";
            break;
        default:
            MessageBox(nullptr, "Unknown hunger state", "Error", MB_ICONERROR);
            break;
    }

    auto lifeEndTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(lifeEndTime - lifeStartTime);

    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration % std::chrono::seconds(1)).count();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count() % 60;
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration).count() % 60;
    hours = std::chrono::duration_cast<std::chrono::hours>(duration).count() % 24;
    auto days = std::chrono::duration_cast<std::chrono::hours>(duration).count() / 24;

    std::cout << "Number of iterations:  " << iteration << std::endl;
    std::cout << "Current Flatten State: " << FlattenState(hungerState, thirstState, agent.left, agent.top) << std::endl;
    std::cout << "Current Left:          " << agent.left << std::endl;
    std::cout << "Current Top:           " << agent.top << std::endl;
    std::cout << "Current Thirst State:  " << thirstStateStr << std::endl;
    std::cout << "Current Hunger State:  " << hungerStateStr << std::endl;
    std::cout << "Current Action:        " << actionStr << std::endl;
    std::cout << "Reward:                " << reward << std::endl;
    std::cout << "Days Lived:            " << days << " days, " << hours << " hours, " << minutes << " minutes, " << seconds << " seconds, and " << milliseconds << " milliseconds" << std::endl;
    std::cout << "Days Without Drinking: " << days << " days, " << hours << " hours, " << minutes << " minutes, " << seconds << " seconds, and " << milliseconds << " milliseconds" << std::endl;
    std::cout << "Days Without Eating:   " << days << " days, " << hours << " hours, " << minutes << " minutes, " << seconds << " seconds, and " << milliseconds << " milliseconds" << std::endl;
    std::cout << "Exploration Rate:      " << exploration_rate << std::endl << std::endl;
}

size_t Environment::Reset()
{
    // numMoveForward = 0;
    numTurnLeft = 0;
    numTurnRight = 0;
    numTurnAround = 0;
    numStatic = 0;
    thirstState = ThirstState::QUENCHED;
    hungerState = HungerState::NEUTRAL;
    currentState = FlattenState(hungerState, thirstState, agent.left, agent.top);
    reward = 0;
    daysLived = 0;
    daysWithoutDrinking = 0;
    daysWithoutEating = 0;
    return currentState;
}

std::tuple<size_t, int, bool> Environment::Step(const size_t action)
{
    switch (action) {
        case Action::MOVE_FORWARD:
            // numMoveForward += 1;
            break;
        case Action::TURN_LEFT:
            numTurnLeft += 1;
            break;
        case Action::TURN_RIGHT:
            numTurnRight += 1;
            break;
        case Action::TURN_AROUND:
            numTurnAround += 1;
            break;
        case Action::STATIC:
            numStatic += 1;
            break;
        default:
            MessageBox(nullptr, "Unknown action", "Error", MB_ICONERROR);
            break;
    }

    if (has_collided_with_water && thirstState != ThirstState::HYDRATED) {
        thirstState = std::min(thirstState + 1, numHungerStates - 1);
        currentState = FlattenState(hungerState, thirstState, agent.left, agent.top);
    } else if (hours >= 3 && thirstState != ThirstState::THIRSTY) {
        thirstState = std::max(thirstState - 1, 0ULL);
        currentState = FlattenState(hungerState, thirstState, agent.left, agent.top);
    }

    if (has_collided_with_food && hungerState != HungerState::FULL) {
        hungerState = std::min(hungerState + 1, numHungerStates - 1);
        currentState = FlattenState(hungerState, thirstState, agent.left, agent.top);
    } else if (hours >= 3 && hungerState != HungerState::HUNGRY) {
        hungerState = std::max(hungerState - 1, 0ULL);
        currentState = FlattenState(hungerState, thirstState, agent.left, agent.top);
    }

    CalculateReward();
    bool done = CheckTermination();

    // if (has_collided_with_food)
    //     daysWithoutEating = 0;
    // else
    //     daysWithoutEating += 1;

    // if (has_collided_with_water)
    //     daysWithoutDrinking = 0;
    // else
    //     daysWithoutDrinking += 1;

    return std::make_tuple(currentState, reward, done);
}

size_t Environment::FlattenState(size_t hungerState, size_t thirstState, LONG left, LONG top) {
    if (!(hungerState < numHungerStates))
        MessageBoxA(nullptr, ("Invalid hunger state. Should be within the range [0, " + std::to_string(numHungerStates) + ")").c_str(), "Error", MB_ICONERROR);
    if (!(thirstState < numThirstStates))
        MessageBox(nullptr, ("Invalid thirst state. Should be within the range [0, " + std::to_string(numThirstStates) + ")").c_str(), "Error", MB_ICONERROR);
    if (!(minLeft <= left && left < numLeftStates) || !(minTop <= top && top < numTopStates))
        MessageBox(nullptr, "Invalid coordinates. Coordinates should be within the specified ranges", "Error", MB_ICONERROR);

    return (((hungerState) * numThirstStates + thirstState) * numLeftStates + static_cast<size_t>(left)) * numTopStates + static_cast<size_t>(top);
}

void Environment::CalculateReward()
{
    reward = 0;

    size_t maxConsecutiveAction = 4;

    if (seenLefts.find(agent.left) != seenLefts.end()) {
        std::cout << "He's been to this left before." << std::endl;
        // Perform some action if the number has been seen before
        // For example, you can break out of the loop
    } else {
        // If the number is not seen before, add it to the set
        seenLefts.insert(agent.left);
        // Perform some action with the new number
        std::cout << "This is a new number." << std::endl;
        // You can also perform other actions here
        reward += 2;
    }

    if (seenTops.find(agent.top) != seenTops.end()) {
        std::cout << "He's been to this left before." << std::endl;
        // Perform some action if the number has been seen before
        // For example, you can break out of the loop
    } else {
        // If the number is not seen before, add it to the set
        seenTops.insert(agent.top);
        // Perform some action with the new number
        std::cout << "This is a new number." << std::endl;
        // You can also perform other actions here
        reward += 2;
    }

    if (daysLived > maxDays)
        reward += 1;
    if (currentState == HungerState::HUNGRY && has_collided_with_food || currentState == HungerState::NEUTRAL && has_collided_with_food)
        reward += 1;
    if (currentState == HungerState::HUNGRY && !has_collided_with_food)
        reward += -1;
    if (currentState == HungerState::HUNGRY && hours >= 3)
        reward += -1;
    if (currentState == HungerState::FULL && has_collided_with_food)
        reward += -1;
    if (has_collided_with_wall)
        reward += -1;
    // if (numMoveForward == maxConsecutiveAction) {
    //     reward += -1;
    //     numMoveForward = 0;
    // }
    if (numTurnLeft == maxConsecutiveAction) {
        reward += -1;
        numTurnLeft = 0;
    }
    if (numTurnRight == maxConsecutiveAction) {
        reward += -1;
        numTurnRight = 0;
    }
    if (numTurnAround == maxConsecutiveAction) {
        reward += -1;
        numTurnAround = 0;
    }
    if (numStatic == maxConsecutiveAction) {
        reward += -1;
        numStatic = 0;
    }
}

bool Environment::CheckTermination()
{
    if (daysLived >= maxDays)
        daysLived = 0;
    
    return daysWithoutEating >= maxDaysWithoutEating || daysLived >= maxDays;
}