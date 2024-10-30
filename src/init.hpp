#pragma once

#include <madrona/sync.hpp>

#include "grid.hpp"
#include "court.hpp"

namespace madsimple {

struct EpisodeManager {
    madrona::AtomicU32 curEpisode;
};

struct WorldInit {
    EpisodeManager *episodeMgr;
    const GridState *grid;
    const CourtState *court; // update initializer
};

}
