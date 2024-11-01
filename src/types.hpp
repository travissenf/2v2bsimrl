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

enum class Action : int32_t {
    Up    = 0,
    Down  = 1,
    Left  = 2,
    Right = 3,
    None,
};

struct GridPos {
    int32_t y;
    int32_t x;
};

// new court position component
struct CourtPos {
    float p1x;
    float p1y;
    float p2x;
    float p2y;
    float p3x;
    float p3y;
    float p4x;
    float p4y;
    float p5x;
    float p5y;
    float p6x;
    float p6y;
    float p7x;
    float p7y;
    float p8x;
    float p8y;
    float p9x;
    float p9y;
    float p10x;
    float p10y;

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