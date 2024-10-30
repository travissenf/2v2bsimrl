#pragma once


// New File, which delcares our Player struct and CourtState struct for internal data management
// This is different than defining archetypes for actually running the madrona simulator
namespace madsimple {

struct Player {
    int32_t id;
    float x;
    float y;
};

struct CourtState {
    Player *players;
    int32_t numPlayers;
};
}
