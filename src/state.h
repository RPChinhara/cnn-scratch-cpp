#pragma once

enum class ThirstState
{
    LEVEL1,
    LEVEL2,
    LEVEL3,
    LEVEL4,
    LEVEL5,
};

enum class HungerState
{
    LEVEL1,
    LEVEL2,
    LEVEL3,
    LEVEL4,
    LEVEL5,
};

enum EnergyState
{
    LEVEL1,
    LEVEL2,
    LEVEL3,
    LEVEL4,
    LEVEL5,
};

enum EmotionState
{
    ANGRY,
    SAD,
    NEUTRAL,
    HAPPY,
};

enum PhysicalHealthState
{
    CRITICAL,
    SICK,
    INJURED,
    HEALTHY,
};

enum class MentalHealthState
{
    CALM,
    RELAXED,
    CONTENT,
    FOCUSED,
    ALERT,
    ENGAGED,
    STRESSED,
    ANXIOUS,
    OVERWHELMED,
    PANICKED,
    NEUTRAL
};
// NOTE: if it reached under certain level, in main.ccp goes back to bee and rest, and recover all the
enum StressState
{
};

enum SleepinessState
{
};

enum SocialConnection
{
};

enum BloodPressureState
{
};

enum ConfidenceState
{
};

enum HygineState
{
};

// TODO: Maybe add 血糖値、血圧?
// TODO I want to add weight, but how?