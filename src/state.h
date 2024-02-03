#pragma once

enum class ThirstState
{
    LEVEL1,
    LEVEL2,
    LEVEL3,
    LEVEL4,
    LEVEL5,
    LEVEL6,
    LEVEL7,
    LEVEL8,
    LEVEL9,
    LEVEL10
};

enum class HungerState
{
    LEVEL1,
    LEVEL2,
    LEVEL3,
    LEVEL4,
    LEVEL5,
    LEVEL6,
    LEVEL7,
    LEVEL8,
    LEVEL9,
    LEVEL10
};

enum EnergyState
{
    LEVEL1,
    LEVEL2,
    LEVEL3,
    LEVEL4,
    LEVEL5,
    LEVEL6,
    LEVEL7,
    LEVEL8,
    LEVEL9,
    LEVEL10
};

enum HealthState
{
    HEALTHY,
    INJURED,
    SICK,
    CRITICAL
};

enum EmotionState
{
    HAPPY,
    SAD,
    ANGRY,
    NEUTRAL
};

enum class MentalState
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

// NOTE:
// 血糖値、血圧、体重
// Physical and mental health? Combine them?
// Emotional state
// Stress state?
// Add Sleepiness, and if it reached under certain level, in main.ccp goes back to bee and rest, and recover all the
// states Do i really need 10 levels for states? 3 is suffice? Ask chatGPT