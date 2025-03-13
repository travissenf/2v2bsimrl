#pragma once
#ifdef gridworld_madrona_mgr_EXPORTS
#define MGR_EXPORT MADRONA_EXPORT
#else
#define MGR_EXPORT MADRONA_IMPORT
#endif

#include <memory>

#include <madrona/py/utils.hpp>
#include <madrona/exec_mode.hpp>

#include "court.hpp"

namespace madsimple {

class Manager {
public:
    // added numPlayers to config to have for later
    struct Config {
        uint32_t maxEpisodeLength;
        madrona::ExecMode execMode;
        uint32_t numWorlds;
        uint32_t numPlayers;
        int gpuID;
    };

    // add initial conditions to manager constructor
    MGR_EXPORT Manager(const Config &cfg, const CourtState &src_court);
    MGR_EXPORT ~Manager();

    MGR_EXPORT void step();

    // new playerTensor
    MGR_EXPORT madrona::py::Tensor playerTensor() const;
    MGR_EXPORT madrona::py::Tensor actionTensor() const;
    MGR_EXPORT madrona::py::Tensor ballTensor() const;
    MGR_EXPORT madrona::py::Tensor heldTensor() const;
    MGR_EXPORT madrona::py::Tensor gameStateTensor() const;
    MGR_EXPORT madrona::py::Tensor playerAttributesTensor() const;
    MGR_EXPORT madrona::py::Tensor choiceTensor() const;
    MGR_EXPORT madrona::py::Tensor foulCallTensor() const;
    MGR_EXPORT madrona::py::Tensor resetTensor() const;

private:
    struct Impl;
    struct CPUImpl;
    struct GPUImpl;

    std::unique_ptr<Impl> impl_;
};

}
