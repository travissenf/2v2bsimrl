#pragma once

#include <madrona/components.hpp>

namespace madsimple {

using madrona::Entity;

enum class ExportID : uint32_t {
    Action,
    CourtPos, // Added a player position archetype for sim
    NumExports,
    BallLoc,
    WhoHolds,
    PassingData,
    Scorecard,
    StaticPlayerAttributes,
    Choice,
};

enum class PlayerDecision : int8_t {
    MOVE = 0,
    SHOOT = 1,
    PASS = 2,
    NOTHING = 3,
};

enum BallStatesPossibilities {
    BALL_IN_LOOSE,
    BALL_IN_PASS,
    BALL_IN_SHOT
};

struct PlayerStatus {
    bool hasBall;
    bool justShot; // TODO: do we need?
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

struct StaticPlayerAttributes {
    float shootingPercentage3Points;
    float shootingPercentageFieldGoal;
    float runningSpeedMph;
}

struct PassingData {
    float i1;
    float i2;
};

struct Scorecard {
    int scoreA;
    int scoreB;
    int quarter;
    float minutesDoneThisQ;
    float secondsDoneThisQ;
};

struct BallReference {
   Entity theBall;
};

struct GameReference {
    Entity theGame;
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
    int8_t whoPassed;
    BallStatesPossibilities ballState;
};

struct AgentList {
    madrona::Entity e[10];
};

struct Agent : public madrona::Archetype<
    Action,
    CourtPos,
    PlayerID,
    PlayerStatus,
    PlayerDecision,
    StaticPlayerAttributes
> {};

struct BallArchetype : public madrona::Archetype<
    BallState,
    BallStatus
> {};

struct GameState : public madrona::Archetype<
    PassingData,
    Scorecard
> {};
}