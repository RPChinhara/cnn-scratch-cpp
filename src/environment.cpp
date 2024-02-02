#include "environment.h"
#include "action.h"
#include "entity.h"
#include "state.h"

#include <chrono>
#include <iostream>
#include <windows.h>

Environment::Environment(const LONG client_width, const LONG client_height)
    : client_width(client_width), client_height(client_height)
{
}

void Environment::Render(const size_t episode, const size_t iteration, Action action, float exploration_rate,
                         Direction direction)
{
    // NOTE: It could be 1.5 ~ 2 seconds per iteration. I set to 1 second for now, but I'm not sure.
    size_t secondsPerIteration = 1;
    secondsLived += secondsPerIteration;
    secondsLivedWithoutDrinking += secondsPerIteration;
    secondsLivedWithoutEating += secondsPerIteration;

    if (secondsLived == 60)
    {
        secondsLived = 0;
        minutesLived += 1;
    }
    if (minutesLived == 60)
    {
        minutesLived = 0;
        hoursLived += 1;
    }
    if (hoursLived == 24)
    {
        hoursLived = 0;
        daysLived += 1;
    }

    if (secondsLivedWithoutDrinking == 60)
    {
        secondsLivedWithoutDrinking = 0;
        minutesLivedWithoutDrinking += 1;
    }
    if (minutesLivedWithoutDrinking == 60)
    {
        minutesLivedWithoutDrinking = 0;
        hoursLivedWithoutDrinking += 1;
    }
    if (hoursLivedWithoutDrinking == 24)
    {
        hoursLivedWithoutDrinking = 0;
        daysLivedWithoutDrinking += 1;
    }

    if (secondsLivedWithoutEating == 60)
    {
        secondsLivedWithoutEating = 0;
        minutesLivedWithoutEating += 1;
    }
    if (minutesLivedWithoutEating == 60)
    {
        minutesLivedWithoutEating = 0;
        hoursLivedWithoutEating += 1;
    }
    if (hoursLivedWithoutEating == 24)
    {
        hoursLivedWithoutEating = 0;
        daysLivedWithoutEating += 1;
    }

    switch (action)
    {
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
    case Action::SLEEP:
        actionStr = "sleep";
        break;
    default:
        MessageBox(nullptr, "Unknown action", "Error", MB_ICONERROR);
        break;
    }

    switch (thirstState)
    {
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

    switch (hungerState)
    {
    case HungerState::LEVEL1:
        hungerStateStr = "level 1";
        break;
    case HungerState::LEVEL2:
        hungerStateStr = "level 2";
        break;
    case HungerState::LEVEL3:
        hungerStateStr = "level 3";
        break;
    case HungerState::LEVEL4:
        hungerStateStr = "level 4";
        break;
    case HungerState::LEVEL5:
        hungerStateStr = "level 5";
        break;
    case HungerState::LEVEL6:
        hungerStateStr = "level 6";
        break;
    case HungerState::LEVEL7:
        hungerStateStr = "level 7";
        break;
    case HungerState::LEVEL8:
        hungerStateStr = "level 8";
        break;
    case HungerState::LEVEL9:
        hungerStateStr = "level 9";
        break;
    case HungerState::LEVEL10:
        hungerStateStr = "level 10";
        break;
    default:
        MessageBox(nullptr, "Unknown hunger state", "Error", MB_ICONERROR);
        break;
    }

    switch (energyState)
    {
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

    if (has_collided_with_water)
        numWaterCollision += 1;
    else if (has_collided_with_food)
        numFoodCollision += 1;
    else if (has_collided_with_agent2)
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

    std::cout << "Episode:                     " << episode << '\n';
    std::cout << "Number of iterations:        " << iteration << '\n';
    std::cout << "Current Flatten State:       "
              << FlattenState(hungerState, thirstState, energyState, agent.left, agent.top) << "/" << numStates << '\n';
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
    std::cout << "Days Lived:                  " << daysLived << " days, " << hoursLived << " hours, " << minutesLived
              << " minutes, " << secondsLived << " seconds" << '\n';
    std::cout << "Days Without Drinking:       " << daysLivedWithoutDrinking << " days, " << hoursLivedWithoutDrinking
              << " hours, " << minutesLivedWithoutDrinking << " minutes, " << secondsLivedWithoutDrinking << " seconds"
              << '\n';
    std::cout << "Days Without Eating:         " << daysLivedWithoutEating << " days, " << hoursLivedWithoutEating
              << " hours, " << minutesLivedWithoutEating << " minutes, " << secondsLivedWithoutEating << " seconds"
              << '\n';
    std::cout << "Exploration Rate:            " << exploration_rate << "\n\n";
}

size_t Environment::Reset()
{
    numWaterCollision = 0;
    numFoodCollision = 0;
    numFriendCollision = 0;
    numWallCollision = 0;

    numWalk = 0;
    numTurnLeft = 0;
    numTurnRight = 0;
    numTurnAround = 0;
    numRun = 0;
    numStatic = 0;

    thirstState = ThirstState::LEVEL5;
    hungerState = HungerState::LEVEL5;
    energyState = EnergyState::LEVEL5;
    currentState = FlattenState(hungerState, thirstState, energyState, agent.left, agent.top);

    reward = 0.0f;

    secondsLived = 0;
    minutesLived = 0;
    hoursLived = 0;
    daysLived = 0;

    secondsLivedWithoutDrinking = 0;
    minutesLivedWithoutDrinking = 0;
    hoursLivedWithoutDrinking = 0;
    daysLivedWithoutDrinking = 0;

    secondsLivedWithoutEating = 0;
    minutesLivedWithoutEating = 0;
    hoursLivedWithoutEating = 0;
    daysLivedWithoutEating = 0;

    energyLevelBelow3 = false;

    return currentState;
}

std::tuple<size_t, float, bool> Environment::Step(Action action)
{
    switch (action)
    {
    case Action::WALK:
        numWalk += 1;
        break;
    case Action::RUN:
        numRun += 1;
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
    case Action::SLEEP:
        numSleep += 1;
        break;
    default:
        MessageBox(nullptr, "Unknown action", "Error", MB_ICONERROR);
        break;
    }

    // TODO: Actions could be diveded into various levels e.g., Eat (Low), Eat (Medium), Eat (High), Exercise (Low).
    size_t thirstStateSizeT = static_cast<size_t>(thirstState);

    if (has_collided_with_water && thirstState != ThirstState::LEVEL10)
    {
        thirstStateSizeT = std::min((thirstStateSizeT + 1), numThirstStates - 1);
        thirstState = static_cast<ThirstState>(thirstStateSizeT);
        currentState = FlattenState(hungerState, thirstState, energyState, agent.left, agent.top);
    }

    if (hoursLivedWithoutDrinking >= 3 && thirstState != ThirstState::LEVEL1)
    {
        thirstStateSizeT = std::max(thirstStateSizeT - 1, 0ULL);
        thirstState = static_cast<ThirstState>(thirstStateSizeT);
        currentState = FlattenState(hungerState, thirstState, energyState, agent.left, agent.top);
        // hours = 0;
    }

    if (action == Action::WALK && numWalk == 200 && thirstState != ThirstState::LEVEL1)
    {
        numWalk = 0;
        thirstStateSizeT = std::max(thirstStateSizeT - 1, 0ULL);
        thirstState = static_cast<ThirstState>(thirstStateSizeT);
        currentState = FlattenState(hungerState, thirstState, energyState, agent.left, agent.top);
    }

    if (action == Action::RUN && numRun == 100 && thirstState != ThirstState::LEVEL1)
    {
        numRun = 0;
        thirstStateSizeT = std::max(thirstStateSizeT - 1, 0ULL);
        thirstState = static_cast<ThirstState>(thirstStateSizeT);
        currentState = FlattenState(hungerState, thirstState, energyState, agent.left, agent.top);
    }

    size_t hungerStateSizeT = static_cast<size_t>(hungerState);

    if (has_collided_with_food && hungerState != HungerState::LEVEL10)
    {
        hungerStateSizeT = std::min((hungerStateSizeT + 1), numHungerStates - 1);
        hungerState = static_cast<HungerState>(hungerStateSizeT);
        currentState = FlattenState(hungerState, thirstState, energyState, agent.left, agent.top);
        // hours = 0;
    }
    // TODO: I think this should be hoursLivedWithoutEating % 3 == 0, and this applies to other places as well.
    // I think above idea is wrong. I think I need to somehow reduce level each time reach multiple of 3 hours or
    // something.
    if (hoursLivedWithoutEating >= 3 && hungerState != HungerState::LEVEL1)
    {
        hungerStateSizeT = std::max(hungerStateSizeT - 1, 0ULL);
        hungerState = static_cast<HungerState>(hungerStateSizeT);
        currentState = FlattenState(hungerState, thirstState, energyState, agent.left, agent.top);
    }

    if (action == Action::WALK && numWalk == 200 && hungerState != HungerState::LEVEL1)
    {
        numWalk = 0;
        hungerStateSizeT = std::max(hungerStateSizeT - 1, 0ULL);
        hungerState = static_cast<HungerState>(hungerStateSizeT);
        currentState = FlattenState(hungerState, thirstState, energyState, agent.left, agent.top);
    }

    if (action == Action::RUN && numRun == 100 && hungerState != HungerState::LEVEL1)
    {
        numRun = 0;
        hungerStateSizeT = std::max(hungerStateSizeT - 1, 0ULL);
        hungerState = static_cast<HungerState>(hungerStateSizeT);
        currentState = FlattenState(hungerState, thirstState, energyState, agent.left, agent.top);
    }

    if (has_collided_with_food && energyState != EnergyState::LEVEL10)
    {
        // TODO: I'm not changing state of the energy here.
        energyState =
            std::min(static_cast<EnergyState>(energyState + 1), static_cast<EnergyState>(numEnergyStates - 1));
        currentState = FlattenState(hungerState, thirstState, energyState, agent.left, agent.top);
    }

    if (hoursLivedWithoutEating >= 3 && energyState != EnergyState::LEVEL1)
    {
        // TODO: I'm not changing state of the energy here.
        energyState = std::max(static_cast<EnergyState>(energyState - 1), static_cast<EnergyState>(0));
        currentState = FlattenState(hungerState, thirstState, energyState, agent.left, agent.top);
    }

    size_t energyStateSizeT = static_cast<size_t>(energyState);

    if (action == Action::WALK && numWalk == 200 && energyState != EnergyState::LEVEL1)
    {
        numWalk = 0;
        energyStateSizeT = std::max(energyStateSizeT - 1, 0ULL);
        energyState = static_cast<EnergyState>(energyStateSizeT);
        currentState = FlattenState(hungerState, thirstState, energyState, agent.left, agent.top);
    }

    if (action == Action::RUN && numRun == 100 && energyState != EnergyState::LEVEL1)
    {
        numRun = 0;
        energyStateSizeT = std::max(energyStateSizeT - 1, 0ULL);
        energyState = static_cast<EnergyState>(energyStateSizeT);
        currentState = FlattenState(hungerState, thirstState, energyState, agent.left, agent.top);
    }

    if (energyState < EnergyState::LEVEL4)
        energyLevelBelow3 = true;
    else if (energyState > EnergyState::LEVEL3)
        energyLevelBelow3 = false;

    // if (action == Action::STATIC && energyState != EnergyState::LEVEL10) {
    //     energyState = std::min(static_cast<EnergyState>(energyState + 1), static_cast<EnergyState>(numEnergyStates -
    //     1)); currentState = FlattenState(hungerState, thirstState, energyState, agent.left, agent.top);
    // }

    CalculateReward(action);
    bool done = CheckTermination();

    if (has_collided_with_water)
    {
        secondsLivedWithoutDrinking = 0;
        minutesLivedWithoutDrinking = 0;
        hoursLivedWithoutDrinking = 0;
        daysLivedWithoutDrinking = 0;
    }

    if (has_collided_with_food)
    {
        secondsLivedWithoutEating = 0;
        minutesLivedWithoutEating = 0;
        hoursLivedWithoutEating = 0;
        daysLivedWithoutEating = 0;
    }

    return std::make_tuple(currentState, reward, done);
}

size_t Environment::FlattenState(HungerState hungerState, ThirstState thirstState, EnergyState energyState, LONG left,
                                 LONG top)
{
    // TODO: Order should be fixed. Like the way parameters are passed. Thirst should come first imo.
    if (!(static_cast<size_t>(hungerState) < numHungerStates))
        MessageBoxA(
            nullptr,
            ("Invalid hunger state. Should be within the range [0, " + std::to_string(numHungerStates) + ")").c_str(),
            "Error", MB_ICONERROR);
    if (!(static_cast<size_t>(thirstState) < numThirstStates))
        MessageBox(
            nullptr,
            ("Invalid thirst state. Should be within the range [0, " + std::to_string(numThirstStates) + ")").c_str(),
            "Error", MB_ICONERROR);
    if (!(static_cast<size_t>(energyState) < numEnergyStates))
        MessageBox(
            nullptr,
            ("Invalid energy state. Should be within the range [0, " + std::to_string(numEnergyStates) + ")").c_str(),
            "Error", MB_ICONERROR);
    if (!(minLeft <= left && left < numLeftStates) || !(minTop <= top && top < numTopStates))
        MessageBox(nullptr, "Invalid coordinates. Coordinates should be within the specified ranges", "Error",
                   MB_ICONERROR);

    // return (((hungerState) * nujmThirstStates + thirstState) * numLeftStates + static_cast<size_t>(left)) *
    // numTopStates + static_cast<size_t>(top);
    return (((energyState * numHungerStates + static_cast<size_t>(hungerState)) * numThirstStates +
             static_cast<size_t>(thirstState)) *
                numLeftStates +
            static_cast<size_t>(left)) *
               numTopStates +
           static_cast<size_t>(top);
}

void Environment::CalculateReward(const Action action)
{
    reward = 0.0f;

    if (std::labs(agent.left - water.left) < 250 && std::labs(agent.top - water.top) < 250)
        reward += 1.2f;

    if (std::labs(agent.left - food.left) < 250 && std::labs(agent.top - food.top) < 250)
        reward += 1.1f;

    if (std::labs(agent.left - agent2.left) < 250 && std::labs(agent.top - agent2.top) < 250)
        reward += 1.0f;

    if (std::labs(agent.left - predator.left) < 250 && std::labs(agent.top - predator.top) < 250)
        reward -= 2.0f;

    if (seenLefts.find(agent.left) != seenLefts.end())
    {
        newLeft = false;
    }
    else
    {
        seenLefts.insert(agent.left);
        newLeft = true;
        reward += 2.2f;
    }

    if (seenTops.find(agent.top) != seenTops.end())
    {
        newTop = false;
    }
    else
    {
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
        reward -= 1.0f;

    if (has_collided_with_food)
        reward += 2.5f;
    if (hungerState == HungerState::LEVEL1 && has_collided_with_food)
        reward += 1.25f;
    if (hungerState == HungerState::LEVEL2 && has_collided_with_food)
        reward += 1.0f;
    if (hungerState == HungerState::LEVEL1 && hoursLivedWithoutEating >= 3)
        reward -= 1.5f;
    if (hungerState == HungerState::LEVEL2 && hoursLivedWithoutEating >= 3)
        reward -= 1.0f;
    if (hungerState == HungerState::LEVEL10 && has_collided_with_food)
        reward -= 2.0f;

    if (energyState == EnergyState::LEVEL1 && action == Action::STATIC)
        reward += 2.0f;
    if (energyState == EnergyState::LEVEL2 && action == Action::STATIC)
        reward += 1.0f;
    if (energyState == EnergyState::LEVEL1 && action == Action::RUN)
        reward -= 2.0f;

    if (has_collided_with_agent2)
        reward += 1.5f;

    if (has_collided_with_wall)
        reward -= 1.5f;
    if (numWallCollision > 1)
        reward -= 2.0f;

    if (has_collided_with_predator)
        reward -= 10.0f;

    // if (numMoveForward == maxConsecutiveAction) {
    //     reward += -1;
    //     numMoveForward = 0;
    // }

    size_t maxConsecutiveAction = 4;
    size_t maxConsecutiveActionTurn = 3;

    if (numTurnLeft == maxConsecutiveActionTurn)
    {
        reward -= 1.0f;
        numTurnLeft = 0;
    }
    if (numTurnRight == maxConsecutiveActionTurn)
    {
        reward -= 1.0f;
        numTurnRight = 0;
    }
    if (numTurnAround == maxConsecutiveActionTurn)
    {
        reward -= 1.0f;
        numTurnAround = 0;
    }
    if (numStatic == maxConsecutiveAction)
    {
        reward -= 1.0f;
        numStatic = 0;
    }
    if (numSleep == maxConsecutiveAction)
    {
        reward -= 1.0f;
        numSleep = 0;
    }
}

bool Environment::CheckTermination()
{
    if (daysLived >= maxDays)
    {
        daysLived = 0;
        return true;
    }

    if (daysLivedWithoutDrinking == maxDaysWithoutDrinking)
        return true;

    if (daysLivedWithoutEating == maxDaysWithoutEating)
        return true;

    if (daysLived == 60 && energyLevelBelow3)
        return true;

    if (has_collided_with_predator)
    {
        // MessageBoxA(NULL, "The agent has been eaten by the predator", "Information", MB_OK | MB_ICONINFORMATION);
        return true;
    }

    return false;
}