#include "environment.h"
#include "action.h"
#include "entity.h"
#include "state.h"

#include <chrono>
#include <iostream>
#include <windows.h>

inline std::chrono::time_point<std::chrono::high_resolution_clock> lifeStartTime;
inline std::chrono::time_point<std::chrono::high_resolution_clock> lifeEndTime;
inline std::chrono::hours::rep hours;
inline std::chrono::hours::rep days;

Environment::Environment(const LONG client_width, const LONG client_height) : client_width(client_width), client_height(client_height) {
}

void Environment::Render(const size_t iteration, Action action, float exploration_rate, Direction direction)
{
    switch (action) {
        case Action::WALK:
            actionStr = "walk";
            break;
        case Action::RUN:
            actionStr = "run";
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
        case ThirstState::LEVEL1:
            thirstStateStr = "level 1";
            break;
        case ThirstState::LEVEL2:
            thirstStateStr = "level 2";
            break;
        case ThirstState::LEVEL3:
            thirstStateStr = "level 3";
            break;
        case ThirstState::LEVEL4:
            thirstStateStr = "level 4";
            break;
        case ThirstState::LEVEL5:
            thirstStateStr = "level 5";
            break;
        case ThirstState::LEVEL6:
            thirstStateStr = "level 6";
            break;
        case ThirstState::LEVEL7:
            thirstStateStr = "level 7";
            break;
        case ThirstState::LEVEL8:
            thirstStateStr = "level 8";
            break;
        case ThirstState::LEVEL9:
            thirstStateStr = "level 9";
            break;
        case ThirstState::LEVEL10:
            thirstStateStr = "level 10";
            break;
        default:
            MessageBox(nullptr, "Unknown thirst state", "Error", MB_ICONERROR);
            break;
    }

    switch (hungerState) {
        case HungerState::HUNGRY:
            hungerStateStr = "hungry";
            break;
        case HungerState::SATISFIED:
            hungerStateStr = "satisfied";
            break;
        case HungerState::FULL:
            hungerStateStr = "full";
            break;
        default:
            MessageBox(nullptr, "Unknown hunger state", "Error", MB_ICONERROR);
            break;
    }

    switch (energyState) {
        case EnergyState::LEVEL1:
            energyStateStr = "level 1";
            break;
        case EnergyState::LEVEL2:
            energyStateStr = "level 2";
            break;
        case EnergyState::LEVEL3:
            energyStateStr = "level 3";
            break;
        case EnergyState::LEVEL4:
            energyStateStr = "level 4";
            break;
        case EnergyState::LEVEL5:
            energyStateStr = "level 5";
            break;
        case EnergyState::LEVEL6:
            energyStateStr = "level 6";
            break;
        case EnergyState::LEVEL7:
            energyStateStr = "level 7";
            break;
        case EnergyState::LEVEL8:
            energyStateStr = "level 8";
            break;
        case EnergyState::LEVEL9:
            energyStateStr = "level 9";
            break;
        case EnergyState::LEVEL10:
            energyStateStr = "level 10";
            break;
        default:
            MessageBox(nullptr, "Unknown energy state", "Error", MB_ICONERROR);
            break;
    }

    auto lifeEndTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(lifeEndTime - lifeStartTime);

    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration % std::chrono::seconds(1)).count();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count() % 60;
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration).count() % 60;
    hours = std::chrono::duration_cast<std::chrono::hours>(duration).count() % 24;
    days = std::chrono::duration_cast<std::chrono::hours>(duration).count() / 24;

    if (has_collided_with_water)
        numWaterCollision += 1;
    else if (has_collided_with_food)
        numFoodCollision += 1;
    else if (has_collided_with_agent_2)
        numFriendCollision += 1;

    std::string currentLeft;
    std::string currentTop;

    if (newLeft)
        currentLeft += "Current Left:                " + std::to_string(agent.left) + " (new)";
    else
        currentLeft += "Current Left:                " + std::to_string(agent.left);
    
    if (newTop)
        currentTop += "Current Top:                 " + std::to_string(agent.top) + " (new)";
    else
        currentTop += "Current Top:                 " + std::to_string(agent.top);

    std::string currentDirection;

    if (direction == Direction::NORTH)
        currentDirection += "north";
    if (direction == Direction::SOUTH)
        currentDirection += "south";
    if (direction == Direction::EAST)
        currentDirection += "east";
    if (direction == Direction::WEST)
        currentDirection += "west";

    std::cout << "Number of iterations:        " << iteration << '\n';
    std::cout << "Current Flatten State:       " << FlattenState(hungerState, thirstState, energyState, agent.left, agent.top) << '\n';
    std::cout << currentLeft << '\n';
    std::cout << currentTop << '\n';
    std::cout << "Current Direction            " << currentDirection << '\n';
    std::cout << "Current Thirst State:        " << thirstStateStr << '\n';
    std::cout << "Current Hunger State:        " << hungerStateStr << '\n';
    std::cout << "Current Energy State:        " << energyStateStr << '\n';
    std::cout << "Current Action:              " << actionStr << '\n';
    std::cout << "Reward:                      " << reward << '\n';
    std::cout << "Number Of Water Collisions:  " << numWaterCollision << '\n';
    std::cout << "Number Of Food Collisions:   " << numFoodCollision << '\n';
    std::cout << "Number Of Friend Collisions: " << numFriendCollision << '\n';
    std::cout << "Days Lived:                  " << days << " days, " << hours << " hours, " << minutes << " minutes, " << seconds << " seconds, and " << milliseconds << " milliseconds" << '\n';
    std::cout << "Days Without Drinking:       " << days << " days, " << hours << " hours, " << minutes << " minutes, " << seconds << " seconds, and " << milliseconds << " milliseconds" << '\n';
    std::cout << "Days Without Eating:         " << days << " days, " << hours << " hours, " << minutes << " minutes, " << seconds << " seconds, and " << milliseconds << " milliseconds" << '\n';
    std::cout << "Exploration Rate:            " << exploration_rate << "\n\n";
}

size_t Environment::Reset()
{
    numWaterCollision = 0;
    numFoodCollision = 0;
    numFriendCollision = 0;
    numWallCollision = 0;
    // numMoveForward = 0;
    numTurnLeft = 0;
    numTurnRight = 0;
    numTurnAround = 0;
    numStatic = 0;
    thirstState = ThirstState::LEVEL5;
    hungerState = HungerState::SATISFIED;
    energyState = EnergyState::LEVEL5;
    currentState = FlattenState(hungerState, thirstState, energyState, agent.left, agent.top);
    reward = 0.0f;
    daysLived = 0;
    daysWithoutDrinking = 0;
    daysWithoutEating = 0;
    energyLevelBelow3 = false;

    return currentState;
}

std::tuple<size_t, float, bool> Environment::Step(Action action)
{
    switch (action) {
        case Action::WALK:
            // numMoveForward += 1;
            break;
        case Action::RUN:
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

    size_t thirstStateSizeT = static_cast<size_t>(thirstState);

    if (has_collided_with_water && thirstState != ThirstState::LEVEL10) {
        thirstStateSizeT = std::min((thirstStateSizeT + 1), numHungerStates - 1);
        thirstState = static_cast<ThirstState>(thirstStateSizeT);
        currentState = FlattenState(hungerState, thirstState, energyState, agent.left, agent.top);
    } else if (hours >= 3 && thirstState != ThirstState::LEVEL1) {
        thirstStateSizeT = std::max(thirstStateSizeT - 1, 0ULL);
        thirstState = static_cast<ThirstState>(thirstStateSizeT);
        currentState = FlattenState(hungerState, thirstState, energyState, agent.left, agent.top);
    }

    if (has_collided_with_food && hungerState != HungerState::FULL) {
        hungerState = std::min(static_cast<HungerState>(hungerState + 1), static_cast<HungerState>(numHungerStates - 1));
        currentState = FlattenState(hungerState, thirstState, energyState, agent.left, agent.top);
    } else if (hours >= 3 && hungerState != HungerState::HUNGRY) {
        hungerState = std::max(static_cast<HungerState>(hungerState - 1), static_cast<HungerState>(0));
        currentState = FlattenState(hungerState, thirstState, energyState, agent.left, agent.top);
    }

    if (has_collided_with_food && energyState != EnergyState::LEVEL10) {
        energyState = std::min(static_cast<EnergyState>(energyState + 1), static_cast<EnergyState>(numEnergyStates - 1));
        currentState = FlattenState(hungerState, thirstState, energyState, agent.left, agent.top);
    }

    if (hours >= 1 && energyState != EnergyState::LEVEL1) {
        energyState = std::max(static_cast<EnergyState>(energyState - 1), static_cast<EnergyState>(0));
        currentState = FlattenState(hungerState, thirstState, energyState, agent.left, agent.top);
    }

    if (energyState < EnergyState::LEVEL4)
        energyLevelBelow3 = true;
    else if (energyState > EnergyState::LEVEL3)
        energyLevelBelow3 = false;

    // if (action == Action::STATIC && energyState != EnergyState::LEVEL10) {
    //     energyState = std::min(static_cast<EnergyState>(energyState + 1), static_cast<EnergyState>(numEnergyStates - 1));
    //     currentState = FlattenState(hungerState, thirstState, energyState, agent.left, agent.top);
    // }

    CalculateReward(action);
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

size_t Environment::FlattenState(HungerState hungerState, ThirstState thirstState, EnergyState energyState, LONG left, LONG top) {
    if (!(static_cast<size_t>(hungerState) < numHungerStates))
        MessageBoxA(nullptr, ("Invalid hunger state. Should be within the range [0, " + std::to_string(numHungerStates) + ")").c_str(), "Error", MB_ICONERROR);
    if (!(static_cast<size_t>(thirstState) < numThirstStates))
        MessageBox(nullptr, ("Invalid thirst state. Should be within the range [0, " + std::to_string(numThirstStates) + ")").c_str(), "Error", MB_ICONERROR);
    if (!(static_cast<size_t>(energyState) < numEnergyStates))
        MessageBox(nullptr, ("Invalid energy state. Should be within the range [0, " + std::to_string(numEnergyStates) + ")").c_str(), "Error", MB_ICONERROR);
    if (!(minLeft <= left && left < numLeftStates) || !(minTop <= top && top < numTopStates))
        MessageBox(nullptr, "Invalid coordinates. Coordinates should be within the specified ranges", "Error", MB_ICONERROR);

    // return (((hungerState) * nujmThirstStates + thirstState) * numLeftStates + static_cast<size_t>(left)) * numTopStates + static_cast<size_t>(top);
    return ((((energyState) * numHungerStates + hungerState) * numThirstStates + static_cast<size_t>(thirstState)) * numLeftStates + static_cast<size_t>(left)) * numTopStates + static_cast<size_t>(top);
}

void Environment::CalculateReward(const Action action)
{
    reward = 0.0f;
    size_t maxConsecutiveAction = 4;

    if (seenLefts.find(agent.left) != seenLefts.end()) {
        newLeft = false;
    } else {
        seenLefts.insert(agent.left);
        newLeft = true;
        reward += 2.2f;
    }

    if (seenTops.find(agent.top) != seenTops.end()) {
        newTop = false;
    } else {
        seenTops.insert(agent.top);
        newTop = true;
        reward += 2.2f;
    }

    if (has_collided_with_wall)
        ++numWallCollision;
    else
        numWallCollision = 0;

    if (daysLived > maxDays)
        reward += 1.0f;

    if (ThirstState::LEVEL1 < thirstState && thirstState < ThirstState::LEVEL5 && has_collided_with_water)
        reward += 1.5f;
    if (ThirstState::LEVEL5 < thirstState && thirstState < ThirstState::LEVEL10 && has_collided_with_water)
        reward += 0.7f;
    if (thirstState == ThirstState::LEVEL10 && has_collided_with_water)
        reward += -1.0f;

    if (has_collided_with_food)
        reward += 2.5f;
    if (hungerState == HungerState::HUNGRY && has_collided_with_food)
        reward += 1.25f;
    if (hungerState == HungerState::SATISFIED && has_collided_with_food)
        reward += 1.0f;
    if (hungerState == HungerState::HUNGRY && hours >= 3)
        reward += -1.0f;
    if (hungerState == HungerState::FULL && has_collided_with_food) {
        reward += -1.0f;

    if (has_collided_with_agent_2)
        reward += 1.5f;
    }

    if (energyState == EnergyState::LEVEL1 && action == Action::STATIC)
        reward += 2.0f;
    if (energyState == EnergyState::LEVEL2 && action == Action::STATIC)
        reward += 1.0f;
    if (energyState == EnergyState::LEVEL1 && action == Action::RUN)
        reward += -2.0f;

    if (has_collided_with_wall)
        reward += -1.5f;
    if (numWallCollision > 1)
        reward += -2.0f;

    // if (numMoveForward == maxConsecutiveAction) {
    //     reward += -1;
    //     numMoveForward = 0;
    // }
    if (numTurnLeft == maxConsecutiveAction) {
        reward += -1.0f;
        numTurnLeft = 0;
    }
    if (numTurnRight == maxConsecutiveAction) {
        reward += -1.0f;
        numTurnRight = 0;
    }
    if (numTurnAround == maxConsecutiveAction) {
        reward += -1.0f;
        numTurnAround = 0;
    }
    if (numStatic == maxConsecutiveAction) {
        reward += -1.0f;
        numStatic = 0;
    }
}

bool Environment::CheckTermination()
{
    if (daysLived >= maxDays)
        daysLived = 0;

    if (days == 60 && energyLevelBelow3)
        return true;
    
    return daysWithoutEating >= maxDaysWithoutEating || daysLived >= maxDays;
}