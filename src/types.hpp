#pragma once

#include <madrona/components.hpp>

namespace madsimple {

enum class ExportID : uint32_t {
    Reset,
    Action,
    GridPos,
    Reward,
    Done,
    CourtPos, // Added a player position archetype for sim
    NumExports,
};

struct Reset {
    int32_t resetNow;
};

// movement action
//      acceleration vector relative to current orientation
//       <1, 0> should be in same direction as current orientation
//      current position
//      current heading angle
//      current velocity
//      current angular velocity (changes heading angle)
//      hasBall
// sim says:
    // if player has ball, move ball according to player
// action space:
//      velocity change
//      angular velocity change
//      dribble
//      pass
//      shoot
//  After basic actions above,
//      rebound?
//      blocks
//      charges

// Start hand crafting policies - 1 person
//      Can hand craft policies to make game look more coherent
//      Given this current state, player 1 should run to the ball, etc.
//     
// 2 people - do more of the simulator

// Logic of how to do actions -- outside of madrona
// can do this in python
// we will have access to positions of players, position of ball, current game state
// Officiating of the game and simulating of the game is inside madrona
// Do the bare minimum and keep on going
// player orientation

// Old action class
// enum class Action : int32_t {
//     Up    = 0,
//     Down  = 1,
//     Left  = 2,
//     Right = 3,
//     None,
// };

struct Action {
    float accel;
    float th;
    float alpha;
};

struct GridPos {
    int32_t y;
    int32_t x;
};

// new court position component
// Can be a court state w/ theta, velocity, ang. velocity (omega)
struct CourtPos {
    float x;
    float y;
    float th;
    float v;
    float om;
    float facing;
};

struct Reward {
    float r;
};

struct Done {
    float episodeDone;
};

struct CurStep {
    uint32_t step;
};

struct Agent : public madrona::Archetype<
    Reset,
    Action,
    GridPos,
    Reward,
    Done,
    CurStep,
    CourtPos
> {};

// new basketball player agent component
// struct PlayerAgent : public madrona::Archetype<
//     Action,
//     CourtPos,
//     CurStep
// > {};

}