#pragma once

enum Action
{
    WALK,
    RUN,
    TURN_LEFT,
    TURN_RIGHT,
    TURN_AROUND,
    STATIC,
    SLEEP // TODO: I think I don't need this. Other way to do this is create hoursWithoutSleeping and if it reached 8 hours, he goes back to his bed and take a sleep.
};