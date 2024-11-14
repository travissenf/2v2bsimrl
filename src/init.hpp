#pragma once

#include <madrona/sync.hpp>

#include "court.hpp"

namespace madsimple {

struct EpisodeManager {
    madrona::AtomicU32 curEpisode;
};

struct WorldInit {
    EpisodeManager *episodeMgr;
    const CourtState *court; // update initializer
};

}
