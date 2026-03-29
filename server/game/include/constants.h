#pragma once

/// Points awarded to other players when one player dies.
const double ALIVE_POINTS = 1.0;
/// Points per controlled hill. This quantity is subtracted when player loses a
/// hill.
extern double POINTS_PER_HILL;
/// Points awarded to player that destroyed a hill.
extern double RAZING_POINTS;

/// Maximum number of unclaimed food on map.
extern int MAX_FOOD_COUNT;
/// Maximum number of turns before game end.
const int MAX_TURN_COUNT = 300;

const int ANT_BATTLE_RADIUS2 = 5;
const int ANT_GATHER_RADIUS2 = 1;
const int ANT_MOVE_RADIUS2 = 1;
