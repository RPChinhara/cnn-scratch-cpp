#include "environment.h"
#include "action.h"
#include "entity.h"
#include "state.h"

#include <chrono>
#include <iostream>
#include <windows.h>

Environment::Environment(const LONG client_width, const LONG client_height, const Agent &agent)
    : client_width(client_width), client_height(client_height)
{
    maxLeft = client_width - agent.width;
    maxTop = client_height - agent.height;

    numLeftStates = (maxLeft - minLeft) + 1;
    numTopStates = (maxTop - minTop) + 1;
    numStates = numPhysicalHealthStates * numEmotionStates * numEnergyStates * numHungerStates * numThirstStates *
                numTopStates * numLeftStates;
}

void Environment::Render(const size_t episode, const size_t iteration, Action action, float exploration_rate,
                         Direction direction, const Agent &agent)
{
    // NOTE: It could be 1.5 ~ 2 seconds per iteration. I set to 1 second for now, but I'm not sure.
    size_t secondsPerIteration = 1;
    secondsLived += secondsPerIteration;
    secondsLivedWithoutDrinking += secondsPerIteration;
    secondsLivedWithoutEating += secondsPerIteration;
    secondsLivedWithoutSocializing += secondsPerIteration;

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

    if (secondsLivedWithoutSocializing == 60)
    {
        secondsLivedWithoutSocializing = 0;
        minutesLivedWithoutSocializing += 1;
    }
    if (minutesLivedWithoutSocializing == 60)
    {
        minutesLivedWithoutSocializing = 0;
        hoursLivedWithoutSocializing += 1;
    }
    if (hoursLivedWithoutSocializing == 24)
    {
        hoursLivedWithoutSocializing = 0;
        daysLivedWithoutSocializing += 1;
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
    default:
        MessageBox(nullptr, "Unknown energy state", "Error", MB_ICONERROR);
        break;
    }

    switch (emotionState)
    {
    case EmotionState::LEVEL1:
        emotionStateStr = "level 1";
        break;
    case EmotionState::LEVEL2:
        emotionStateStr = "level 2";
        break;
    case EmotionState::LEVEL3:
        emotionStateStr = "level 3";
        break;
    case EmotionState::LEVEL4:
        emotionStateStr = "level 4";
        break;
    default:
        MessageBox(nullptr, "Unknown emotion state", "Error", MB_ICONERROR);
        break;
    }

    switch (physicalHealthState)
    {
    case PhysicalHealthState::LEVEL1:
        physicalHealthStateStr = "level 1";
        break;
    case PhysicalHealthState::LEVEL2:
        physicalHealthStateStr = "level 2";
        break;
    case PhysicalHealthState::LEVEL3:
        physicalHealthStateStr = "level 3";
        break;
    case PhysicalHealthState::LEVEL4:
        physicalHealthStateStr = "level 4";
        break;
    default:
        MessageBox(nullptr, "Unknown physical health state", "Error", MB_ICONERROR);
        break;
    }

    if (agent.has_collided_with_water)
        numWaterCollision += 1;
    else if (agent.has_collided_with_food)
        numFoodCollision += 1;
    else if (agent.has_collided_with_agent2)
        numFriendCollision += 1;

    std::string currentLeft;
    std::string currentTop;

    if (newLeft)
        currentLeft += "Current Left:                  " + std::to_string(agent.position.left) + " (new)";
    else
        currentLeft += "Current Left:                  " + std::to_string(agent.position.left);

    if (newTop)
        currentTop += "Current Top:                   " + std::to_string(agent.position.top) + " (new)";
    else
        currentTop += "Current Top:                   " + std::to_string(agent.position.top);

    std::string currentDirection;

    if (direction == Direction::NORTH)
        currentDirection += "north";
    if (direction == Direction::SOUTH)
        currentDirection += "south";
    if (direction == Direction::EAST)
        currentDirection += "east";
    if (direction == Direction::WEST)
        currentDirection += "west";

    std::cout << "Episode:                       " << episode << '\n';
    std::cout << "Number of iterations:          " << iteration << '\n';
    std::cout << "Current Flatten State:         "
              << FlattenState(agent.position.left, agent.position.top, thirstState, hungerState, energyState,
                              emotionState)
              << "/" << numStates << '\n';
    std::cout << currentLeft << '\n';
    std::cout << currentTop << '\n';
    std::cout << "Current Direction              " << currentDirection << '\n';
    std::cout << "Current Thirst State:          " << thirstStateStr << '\n';
    std::cout << "Current Hunger State:          " << hungerStateStr << '\n';
    std::cout << "Current Energy State:          " << energyStateStr << '\n';
    std::cout << "Current Emotion State:         " << emotionStateStr << '\n';
    std::cout << "Current Physical Health State: " << physicalHealthStateStr << '\n';
    std::cout << "Current Action:                " << actionStr << '\n';
    std::cout << "Reward:                        " << reward << '\n';
    std::cout << "Number Of Water Collisions:    " << numWaterCollision << '\n';
    std::cout << "Number Of Food Collisions:     " << numFoodCollision << '\n';
    std::cout << "Number Of Friend Collisions:   " << numFriendCollision << '\n';
    std::cout << "Days Lived:                    " << daysLived << " days, " << hoursLived << " hours, " << minutesLived
              << " minutes, " << secondsLived << " seconds" << '\n';
    std::cout << "Days Without Drinking:         " << daysLivedWithoutDrinking << " days, " << hoursLivedWithoutDrinking
              << " hours, " << minutesLivedWithoutDrinking << " minutes, " << secondsLivedWithoutDrinking << " seconds"
              << '\n';
    std::cout << "Days Without Eating:           " << daysLivedWithoutEating << " days, " << hoursLivedWithoutEating
              << " hours, " << minutesLivedWithoutEating << " minutes, " << secondsLivedWithoutEating << " seconds"
              << '\n';
    std::cout << "Days Without Socializing:      " << daysLivedWithoutSocializing << " days, "
              << hoursLivedWithoutSocializing << " hours, " << minutesLivedWithoutSocializing << " minutes, "
              << secondsLivedWithoutSocializing << " seconds" << '\n';
    std::cout << "Exploration Rate:              " << exploration_rate << "\n";
}

size_t Environment::Reset(const Agent &agent)
{
    prevHasCollidedWithWater = false;
    prevHasCollidedWithFood = false;

    numWaterCollision = 0;
    numFoodCollision = 0;
    numFriendCollision = 0;
    numFriendCollisionWhileHappy = 0;
    numWallCollision = 0;

    numWalk = 0;
    numTurnLeft = 0;
    numTurnRight = 0;
    numTurnAround = 0;
    numRun = 0;
    numStatic = 0;

    thirstState = ThirstState::LEVEL3;
    hungerState = HungerState::LEVEL3;
    energyState = EnergyState::LEVEL3;
    emotionState = EmotionState::LEVEL1;
    physicalHealthState = PhysicalHealthState::LEVEL1;

    currentState =
        FlattenState(agent.position.left, agent.position.top, thirstState, hungerState, energyState, emotionState);

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

    secondsLivedWithoutSocializing = 0;
    minutesLivedWithoutSocializing = 0;
    hoursLivedWithoutSocializing = 0;
    daysLivedWithoutSocializing = 0;

    energyLevelBelow3 = false;

    seenLefts.clear();
    seenTops.clear();

    return currentState;
}

std::tuple<size_t, float, bool> Environment::Step(Action action, const Agent &agent)
{
    if (agent.has_collided_with_agent2 && emotionState == EmotionState::LEVEL4)
        ++numFriendCollisionWhileHappy;

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

    size_t thirstStateSizeT = static_cast<size_t>(thirstState);

    if (agent.has_collided_with_water && thirstState != ThirstState::LEVEL5)
    {
        // TODO: Is this really the best way to decrease/increase the elements of the enum?
        thirstStateSizeT = std::min((thirstStateSizeT + 1), numThirstStates - 1);
        thirstState = static_cast<ThirstState>(thirstStateSizeT);
        currentState =
            FlattenState(agent.position.left, agent.position.top, thirstState, hungerState, energyState, emotionState);
    }

    if (hoursLivedWithoutDrinking >= 3 && thirstState != ThirstState::LEVEL1)
    {
        thirstStateSizeT = std::max(thirstStateSizeT - 1, 0ULL);
        thirstState = static_cast<ThirstState>(thirstStateSizeT);
        currentState =
            FlattenState(agent.position.left, agent.position.top, thirstState, hungerState, energyState, emotionState);
        // hoursLivedWithoutDrinking = 0; TODO: Do I have to reset it to 0?
    }
    // TODO: I think I could refactor thirst, hunger, and energy parts as these are doing same thing.
    // IDEA: I think 200 and 100 is too small, he get exausted too early imo.
    if (action == Action::WALK && numWalk == 700 && thirstState != ThirstState::LEVEL1)
    {
        thirstStateSizeT = std::max(thirstStateSizeT - 1, 0ULL);
        thirstState = static_cast<ThirstState>(thirstStateSizeT);
        currentState =
            FlattenState(agent.position.left, agent.position.top, thirstState, hungerState, energyState, emotionState);
    }

    if (action == Action::RUN && numRun == 600 && thirstState != ThirstState::LEVEL1)
    {
        thirstStateSizeT = std::max(thirstStateSizeT - 1, 0ULL);
        thirstState = static_cast<ThirstState>(thirstStateSizeT);
        currentState =
            FlattenState(agent.position.left, agent.position.top, thirstState, hungerState, energyState, emotionState);
    }

    size_t hungerStateSizeT = static_cast<size_t>(hungerState);

    if (agent.has_collided_with_food && hungerState != HungerState::LEVEL5)
    {
        hungerStateSizeT = std::min((hungerStateSizeT + 1), numHungerStates - 1);
        hungerState = static_cast<HungerState>(hungerStateSizeT);
        currentState =
            FlattenState(agent.position.left, agent.position.top, thirstState, hungerState, energyState, emotionState);
        // hours = 0;
    }
    // TODO: I think this should be hoursLivedWithoutEating % 3 == 0, and this applies to other places as well.
    // I think above idea is wrong. I think I need to somehow reduce level each time reach multiple of 3 hours or
    // something.
    if (hoursLivedWithoutEating >= 3 && hungerState != HungerState::LEVEL1)
    {
        hungerStateSizeT = std::max(hungerStateSizeT - 1, 0ULL);
        hungerState = static_cast<HungerState>(hungerStateSizeT);
        currentState =
            FlattenState(agent.position.left, agent.position.top, thirstState, hungerState, energyState, emotionState);
    }

    if (action == Action::WALK && numWalk == 700 && hungerState != HungerState::LEVEL1)
    {
        hungerStateSizeT = std::max(hungerStateSizeT - 1, 0ULL);
        hungerState = static_cast<HungerState>(hungerStateSizeT);
        currentState =
            FlattenState(agent.position.left, agent.position.top, thirstState, hungerState, energyState, emotionState);
    }

    if (action == Action::RUN && numRun == 600 && hungerState != HungerState::LEVEL1)
    {
        hungerStateSizeT = std::max(hungerStateSizeT - 1, 0ULL);
        hungerState = static_cast<HungerState>(hungerStateSizeT);
        currentState =
            FlattenState(agent.position.left, agent.position.top, thirstState, hungerState, energyState, emotionState);
    }

    if (agent.has_collided_with_food && energyState != EnergyState::LEVEL5)
    {
        // TODO: I'm not changing state of the energy here.
        energyState =
            std::min(static_cast<EnergyState>(energyState + 1), static_cast<EnergyState>(numEnergyStates - 1));
        currentState =
            FlattenState(agent.position.left, agent.position.top, thirstState, hungerState, energyState, emotionState);
    }

    if (hoursLivedWithoutEating >= 3 && energyState != EnergyState::LEVEL1)
    {
        // TODO: I'm not changing state of the energy here.
        energyState = std::max(static_cast<EnergyState>(energyState - 1), static_cast<EnergyState>(0));
        currentState =
            FlattenState(agent.position.left, agent.position.top, thirstState, hungerState, energyState, emotionState);
    }

    size_t energyStateSizeT = static_cast<size_t>(energyState);

    if (action == Action::WALK && numWalk == 700 && energyState != EnergyState::LEVEL1)
    {
        energyStateSizeT = std::max(energyStateSizeT - 1, 0ULL);
        energyState = static_cast<EnergyState>(energyStateSizeT);
        currentState =
            FlattenState(agent.position.left, agent.position.top, thirstState, hungerState, energyState, emotionState);
    }

    if (action == Action::RUN && numRun == 600 && energyState != EnergyState::LEVEL1)
    {
        energyStateSizeT = std::max(energyStateSizeT - 1, 0ULL);
        energyState = static_cast<EnergyState>(energyStateSizeT);
        currentState =
            FlattenState(agent.position.left, agent.position.top, thirstState, hungerState, energyState, emotionState);
    }

    size_t emotionStateSizeT = static_cast<size_t>(emotionState);

    if (agent.has_collided_with_food && emotionState != EmotionState::LEVEL4 ||
        agent.has_collided_with_agent2 && emotionState != EmotionState::LEVEL4)
    {
        emotionStateSizeT = std::min((emotionStateSizeT + 1), numEmotionStates - 1);
        emotionState = static_cast<EmotionState>(emotionStateSizeT);
        currentState =
            FlattenState(agent.position.left, agent.position.top, thirstState, hungerState, energyState, emotionState);
    }

    if (hoursLivedWithoutEating >= 8 && emotionState != EmotionState::LEVEL1)
    {
        emotionStateSizeT = std::max(emotionStateSizeT - 1, 0ULL);
        emotionState = static_cast<EmotionState>(emotionStateSizeT);
        currentState =
            FlattenState(agent.position.left, agent.position.top, thirstState, hungerState, energyState, emotionState);
    }

    if (hoursLivedWithoutSocializing >= 8 && emotionState != EmotionState::LEVEL1)
    {
        emotionStateSizeT = std::max(emotionStateSizeT - 1, 0ULL);
        emotionState = static_cast<EmotionState>(emotionStateSizeT);
        currentState =
            FlattenState(agent.position.left, agent.position.top, thirstState, hungerState, energyState, emotionState);
    }

    if (numFriendCollisionWhileHappy >= 3 && emotionState != EmotionState::LEVEL1)
    {
        emotionStateSizeT = std::max(emotionStateSizeT - 1, 0ULL);
        emotionState = static_cast<EmotionState>(emotionStateSizeT);
        currentState =
            FlattenState(agent.position.left, agent.position.top, thirstState, hungerState, energyState, emotionState);
    }

    size_t physicalHealthStateSizeT = static_cast<size_t>(physicalHealthState);

    if (agent.has_collided_with_predator && physicalHealthState != PhysicalHealthState::LEVEL1)
    {
        physicalHealthStateSizeT = std::min((physicalHealthStateSizeT + 1), numPhysicalHealthStates - 1);
        physicalHealthState = static_cast<PhysicalHealthState>(physicalHealthStateSizeT);
        currentState =
            FlattenState(agent.position.left, agent.position.top, thirstState, hungerState, energyState, emotionState);
    }

    // TODO: Increase level of emotion state when he kill the predators, and decrease when opposite result occured.

    if (energyState < EnergyState::LEVEL4)
        energyLevelBelow3 = true;
    else if (energyState > EnergyState::LEVEL3)
        energyLevelBelow3 = false;

    if (numWalk == 700 || numRun == 600)
    {
        numWalk = 0;
        numRun = 0;
    }

    // if (action == Action::STATIC && energyState != EnergyState::LEVEL10) {
    //     energyState = std::min(static_cast<EnergyState>(energyState + 1), static_cast<EnergyState>(numEnergyStates -
    //     1)); currentState = FlattenState(hungerState, thirstState, energyState, emotionState, agent.left, agent.top);
    // }

    CalculateReward(action, agent);
    bool done = CheckTermination(agent);

    if (agent.has_collided_with_water)
    {
        secondsLivedWithoutDrinking = 0;
        minutesLivedWithoutDrinking = 0;
        hoursLivedWithoutDrinking = 0;
        daysLivedWithoutDrinking = 0;
    }

    if (agent.has_collided_with_food)
    {
        secondsLivedWithoutEating = 0;
        minutesLivedWithoutEating = 0;
        hoursLivedWithoutEating = 0;
        daysLivedWithoutEating = 0;
    }

    if (agent.has_collided_with_agent2)
    {
        secondsLivedWithoutSocializing = 0;
        minutesLivedWithoutSocializing = 0;
        hoursLivedWithoutSocializing = 0;
        daysLivedWithoutSocializing = 0;
    }

    return std::make_tuple(currentState, reward, done);
}

size_t Environment::FlattenState(LONG left, LONG top, ThirstState thirstState, HungerState hungerState,
                                 EnergyState energyState, EmotionState emotionState)
{
    if (!(minLeft <= left && left < numLeftStates) || !(minTop <= top && top < numTopStates))
        MessageBox(nullptr, "Invalid coordinates. Coordinates should be within the specified ranges", "Error",
                   MB_ICONERROR);
    if (!(static_cast<size_t>(thirstState) < numThirstStates))
        MessageBox(
            nullptr,
            ("Invalid thirst state. Should be within the range [0, " + std::to_string(numThirstStates) + ")").c_str(),
            "Error", MB_ICONERROR);
    if (!(static_cast<size_t>(hungerState) < numHungerStates))
        MessageBoxA(
            nullptr,
            ("Invalid hunger state. Should be within the range [0, " + std::to_string(numHungerStates) + ")").c_str(),
            "Error", MB_ICONERROR);
    if (!(static_cast<size_t>(energyState) < numEnergyStates))
        MessageBox(
            nullptr,
            ("Invalid energy state. Should be within the range [0, " + std::to_string(numEnergyStates) + ")").c_str(),
            "Error", MB_ICONERROR);

    // return (((hungerState) * nujmThirstStates + thirstState) * numLeftStates + static_cast<size_t>(left)) *
    // numTopStates + static_cast<size_t>(top);
    return ((((static_cast<size_t>(emotionState) * numEnergyStates + energyState) * numHungerStates +
              static_cast<size_t>(hungerState)) *
                 numThirstStates +
             static_cast<size_t>(thirstState)) *
                numTopStates +
            static_cast<size_t>(top)) *
               numLeftStates +
           static_cast<size_t>(left);
}

void Environment::CalculateReward(const Action action, const Agent &agent)
{
    reward = 0.0f;

    // TODO: I don't know, but pass vector of entity in here as there are various enties's postions has been accessed.
    if (std::labs(agent.position.left - water.left) < 250 && std::labs(agent.position.top - water.top) < 250)
        reward += 1.2f;

    if (std::labs(agent.position.left - food.left) < 250 && std::labs(agent.position.top - food.top) < 250)
        reward += 1.1f;

    if (std::labs(agent.position.left - agent2.left) < 250 && std::labs(agent.position.top - agent2.top) < 250)
        reward += 1.0f;

    if (std::labs(agent.position.left - predator.left) < 250 && std::labs(agent.position.top - predator.top) < 250)
        reward -= 2.0f;

    if (seenLefts.find(agent.position.left) != seenLefts.end())
    {
        newLeft = false;
    }
    else
    {
        seenLefts.insert(agent.position.left);
        newLeft = true;
        reward += 2.2f;
    }

    if (seenTops.find(agent.position.top) != seenTops.end())
    {
        newTop = false;
    }
    else
    {
        seenTops.insert(agent.position.top);
        newTop = true;
        reward += 2.2f;
    }

    // TODO: I think I need to make numWaterCollision, numFoodCollision because otherwise he will drink or eat forever
    // which lead to death in irl.

    // TODO: This is not considering sequentially collided with the wall. It's just incrementing whenever he collided
    // which is not corerct way to do it?
    if (agent.has_collided_with_wall)
        ++numWallCollision;
    else
        numWallCollision = 0;

    if (daysLived > maxDays)
        reward += 1.0f;

    // TODO: Should be thirstState > ThirstState::LEVEL1
    if (ThirstState::LEVEL1 < thirstState && thirstState < ThirstState::LEVEL5 && agent.has_collided_with_water)
        reward += 1.5f;
    if (ThirstState::LEVEL5 < thirstState && thirstState < ThirstState::LEVEL5 && agent.has_collided_with_water)
        reward += 0.7f;
    if (thirstState == ThirstState::LEVEL5 && agent.has_collided_with_water)
        reward -= 3.0f;
    if (agent.has_collided_with_water && prevHasCollidedWithWater && thirstState == ThirstState::LEVEL5)
        reward -= 3.0f;

    if (agent.has_collided_with_food)
        reward += 2.5f;
    if (hungerState == HungerState::LEVEL1 && agent.has_collided_with_food)
        reward += 1.25f;
    if (hungerState == HungerState::LEVEL2 && agent.has_collided_with_food)
        reward += 1.0f;
    if (hungerState == HungerState::LEVEL1 && hoursLivedWithoutEating >= 3)
        reward -= 1.5f;
    if (hungerState == HungerState::LEVEL2 && hoursLivedWithoutEating >= 3)
        reward -= 1.0f;
    if (hungerState == HungerState::LEVEL5 && agent.has_collided_with_food)
        reward -= 3.0f;
    if (agent.has_collided_with_food && prevHasCollidedWithFood && hungerState == HungerState::LEVEL5)
        reward -= 3.0f;

    if (energyState == EnergyState::LEVEL1 && action == Action::STATIC)
        reward += 2.0f;
    if (energyState == EnergyState::LEVEL2 && action == Action::STATIC)
        reward += 1.0f;
    if (energyState == EnergyState::LEVEL1 && action == Action::RUN)
        reward -= 2.0f;

    if (agent.has_collided_with_agent2)
        reward += 1.5f;

    if (agent.has_collided_with_wall)
        reward -= 1.5f;
    if (numWallCollision > 1)
        reward -= 2.0f;

    if (agent.has_collided_with_predator)
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

    if (agent.has_collided_with_water)
        prevHasCollidedWithWater = true;
    else
        prevHasCollidedWithWater = false;

    if (agent.has_collided_with_food)
        prevHasCollidedWithFood = true;
    else
        prevHasCollidedWithFood = false;
}

bool Environment::CheckTermination(const Agent &agent)
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

    if (agent.has_collided_with_predator && physicalHealthState == PhysicalHealthState::LEVEL1)
    {
        // MessageBoxA(NULL, "The agent has been eaten by the predator", "Information", MB_OK | MB_ICONINFORMATION);
        return true;
    }

    return false;
}