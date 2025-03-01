#pragma once

#include <madrona/components.hpp>
#include "consts.hpp"

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
    CalledFoul
};

enum class PlayerDecision : int8_t {
    MOVE = 0,
    SHOOT = 1,
    PASS = 2,
    NOTHING = 3,
};

enum class FoulID : int8_t {
    NO_CALL = 0,
    BLOCK = 1,
    CHARGE = 2,
    PUSH = 3,
};

struct PlayerStatus {
    bool hasBall;
    bool justShot; // TODO: do we need?
    int8_t pointsOnMake;
};

struct Action {
    float vdes;
    float thdes;
    float omdes;
    float pass_th;
    float pass_v;
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
};

struct Scorecard {
    int16_t score1;
    int16_t score2;
    int16_t quarter;
    int16_t ticksElapsed;
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

enum BallStatesPossibilities {
    BALL_IN_LOOSE = 0,
    BALL_IN_PASS = 1,
    BALL_IN_SHOT = 2,
    T1_NEED_TO_INBOUND = 3,
    T2_NEED_TO_INBOUND = 4, 
    BALL_IS_HELD = 5
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
    madrona::Entity e[ACTIVE_PLAYERS];
};

struct Agent : public madrona::Archetype<
    Action,
    CourtPos,
    PlayerID,
    PlayerStatus,
    PlayerDecision,
    FoulID,
    StaticPlayerAttributes
> {};

struct BallArchetype : public madrona::Archetype<
    BallState,
    BallStatus
> {};

struct GameState : public madrona::Archetype<
    Scorecard
> {};
}