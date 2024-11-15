#pragma once

#include <madrona/components.hpp>

namespace madsimple {

using madrona::Entity;

enum class ExportID : uint32_t {
    Action,
    CourtPos, // Added a player position archetype for sim
    NumExports,
    // PlayerStatus,
    BallLoc,
    WhoHolds,
    Choice,
};

enum PlayerDecision {
    MOVE,
    SHOOT,
    PASS
};

enum BallStatesPossibilities {
    BALL_IN_LOOSE,
    BALL_IN_PASS,
    BALL_IN_SHOT
};

struct PlayerStatus {
    bool hasBall;
    bool justShot; // TODO: do we need?
    PlayerDecision playerDecision;
};

struct Action {
    float vdes;
    float thdes;
    float omdes;
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

struct BallReference {
   Entity theBall;
};

struct PlayerID {
    int8_t id;
};

struct BallState {
    float x;
    float y;
    float th;    
    float v;
};

struct BallStatus {
    int8_t heldBy;
    int8_t whoShot;
    BallStatesPossibilities ballState;
};

struct AgentList {
    madrona::Entity e[10];
};

struct Agent : public madrona::Archetype<
    Action,
    // CurStep,  // make singleton
    CourtPos,
    PlayerID,
    PlayerStatus
> {};

struct BallArchetype : public madrona::Archetype<
    BallState,
    BallStatus
> {};
}