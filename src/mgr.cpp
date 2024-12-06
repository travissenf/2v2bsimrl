#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/mw_cpu.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

using namespace madrona;
using namespace madrona::py;

namespace madsimple {

struct Manager::Impl {
    Config cfg;
    EpisodeManager *episodeMgr;

    // Added courtData structure, which contains number of players, and array of players and their locations
    CourtState *courtData;

    // Added court_state ot constructor, which gives input to courtData
    inline Impl(const Config &c,
                EpisodeManager *ep_mgr,
                CourtState *court_state)
        : cfg(c),
          episodeMgr(ep_mgr),
          courtData(court_state)
    {}

    inline virtual ~Impl() {}

    virtual void run() = 0;
    virtual Tensor exportTensor(ExportID slot, TensorElementType type,
                                Span<const int64_t> dims) = 0;

    // Add CourtState to constructor
    static inline Impl * init(const Config &cfg, const CourtState &src_players);
};

struct Manager::CPUImpl final : Manager::Impl {
    using ExecT = TaskGraphExecutor<Engine, Sim, Sim::Config, WorldInit>;
    ExecT cpuExec;

    // Add courtData to constructor
    inline CPUImpl(const Manager::Config &mgr_cfg,
                   const Sim::Config &sim_cfg,
                   EpisodeManager *episode_mgr,
                   CourtState *court_data,
                   WorldInit *world_inits)
        : Impl(mgr_cfg, episode_mgr, court_data),
          cpuExec({
                  .numWorlds = mgr_cfg.numWorlds,
                  .numExportedBuffers = (uint32_t)ExportID::NumExports,
              }, sim_cfg, world_inits, 1)
    {}

    // Free courtData
    inline virtual ~CPUImpl() final {
        delete episodeMgr;
        free(courtData);
    }

    inline virtual void run() final { cpuExec.run(); }
    
    inline virtual Tensor exportTensor(ExportID slot,
                                       TensorElementType type,
                                       Span<const int64_t> dims) final
    {
        void *dev_ptr = cpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, Optional<int>::none());
    }
};

// Updated this GPU support, however unsure if this runs on CUDA yet
#ifdef MADRONA_CUDA_SUPPORT
struct Manager::GPUImpl final : Manager::Impl {
    MWCudaExecutor gpuExec;
    MWCudaLaunchGraph stepGraph;

    inline GPUImpl(CUcontext cu_ctx,
                   const Manager::Config &mgr_cfg,
                   const Sim::Config &sim_cfg,
                   EpisodeManager *episode_mgr,
                   CourtState *court_data,
                   WorldInit *world_inits)
        : Impl(mgr_cfg, episode_mgr, court_data),
          gpuExec({
                  .worldInitPtr = world_inits,
                  .numWorldInitBytes = sizeof(WorldInit),
                  .userConfigPtr = (void *)&sim_cfg,
                  .numUserConfigBytes = sizeof(Sim::Config),
                  .numWorldDataBytes = sizeof(Sim),
                  .worldDataAlignment = alignof(Sim),
                  .numWorlds = mgr_cfg.numWorlds,
                  .numTaskGraphs = 1,
                  .numExportedBuffers = (uint32_t)ExportID::NumExports, 
              }, {
                  { SIMPLE_SRC_LIST },
                  { SIMPLE_COMPILE_FLAGS },
                  CompileConfig::OptMode::LTO,
              }, cu_ctx),
          stepGraph(gpuExec.buildLaunchGraph(0))
          
    {}

    inline virtual ~GPUImpl() final {
        REQ_CUDA(cudaFree(episodeMgr));
        REQ_CUDA(cudaFree(courtData));
    }

    inline virtual void run() final { gpuExec.run(stepGraph); }
    
    virtual inline Tensor exportTensor(ExportID slot, TensorElementType type,
                                       Span<const int64_t> dims) final
    {
        void *dev_ptr = gpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, cfg.gpuID);
    }
};
#endif

// Added CourtState to world initialization
static HeapArray<WorldInit> setupWorldInitData(int64_t num_worlds,
                                               EpisodeManager *episode_mgr,
                                               const CourtState *court)
{
    HeapArray<WorldInit> world_inits(num_worlds);

    for (int64_t i = 0; i < num_worlds; i++) {
        world_inits[i] = WorldInit {
            episode_mgr,
            court,
        };
    }

    return world_inits;
}

// Added CourtState to this
Manager::Impl * Manager::Impl::init(const Config &cfg,
                                    const CourtState &src_court)
{

    Sim::Config sim_cfg {
        .maxEpisodeLength = cfg.maxEpisodeLength,
        .enableViewer = false,
    };

    switch (cfg.execMode) {
    case ExecMode::CPU: {
        EpisodeManager *episode_mgr = new EpisodeManager { 0 };


        // Block of code that mallocs the CourtState object, plus the players array it points to
        uint64_t player_bytes = sizeof(Player) * src_court.numPlayers;

        auto *court_data =
            (char *)malloc(sizeof(CourtState) + player_bytes);
        Player *cpu_player_data = (Player *)(court_data + sizeof(CourtState));

        CourtState *cpu_court = (CourtState *)court_data;

        *cpu_court = CourtState {
            .players = cpu_player_data,
            .numPlayers = src_court.numPlayers,
        };

        // cpu_court now contains all of our input data

        memcpy(cpu_player_data, src_court.players, player_bytes);

        HeapArray<WorldInit> world_inits = setupWorldInitData(cfg.numWorlds,
            episode_mgr, cpu_court);

        return new CPUImpl(cfg, sim_cfg, episode_mgr, cpu_court, world_inits.data());
    } break;
    case ExecMode::CUDA: {
        // I have not implemented in the CUDA for this section yet
#ifndef MADRONA_CUDA_SUPPORT
        FATAL("CUDA support not compiled in!");
#else
        CUcontext cu_ctx = MWCudaExecutor::initCUDA(cfg.gpuID);

        EpisodeManager *episode_mgr = 
            (EpisodeManager *)cu::allocGPU(sizeof(EpisodeManager));
        // Set the current episode count to 0
        REQ_CUDA(cudaMemset(episode_mgr, 0, sizeof(EpisodeManager)));


        // Block of code that mallocs the CourtState object, plus the players array it points to
        uint64_t player_bytes = sizeof(Player) * src_court.numPlayers;

        auto *court_data =
            (char *)malloc(sizeof(CourtState) + player_bytes);
        Player *cpu_player_data = (Player *)(court_data + sizeof(CourtState));

        CourtState *cpu_court = (CourtState *)court_data;

        *cpu_court = CourtState {
            .players = cpu_player_data,
            .numPlayers = src_court.numPlayers,
        };


        HeapArray<WorldInit> world_inits = setupWorldInitData(cfg.numWorlds,
            episode_mgr, cpu_court);

        return new GPUImpl(cu_ctx, cfg, sim_cfg, episode_mgr, cpu_court,
                           world_inits.data());
#endif
    } break;
    default: return nullptr;
    }
}

// Added initial conditions to manager
Manager::Manager(const Config &cfg,
                 const CourtState &src_court)
    : impl_(Impl::init(cfg, src_court))
{}

Manager::~Manager() {}

void Manager::step()
{
    impl_->run();
}

// Added new tensor playerTensor, that theoretically will hold [numWorlds, numPlayers, location] (unsure about this implementation)
Tensor Manager::playerTensor() const
{
     return impl_->exportTensor(ExportID::CourtPos, TensorElementType::Float32,
        {impl_->cfg.numWorlds, impl_->cfg.numPlayers, 6});
}

Tensor Manager::actionTensor() const
{
    return impl_->exportTensor(ExportID::Action, TensorElementType::Float32,
        {impl_->cfg.numWorlds, impl_->cfg.numPlayers, 3});
}

Tensor Manager::ballTensor() const
{
    return impl_->exportTensor(ExportID::BallLoc, TensorElementType::Float32,
        {impl_->cfg.numWorlds, 4});
}

Tensor Manager::heldTensor() const
{
    return impl_->exportTensor(ExportID::WhoHolds, TensorElementType::Int8,
        {impl_->cfg.numWorlds, 2});
}

Tensor Manager::passingDataTensor() const
{
    return impl_->exportTensor(ExportID::PassingData, TensorElementType::Float32,
        {impl_->cfg.numWorlds, 2});
}

Tensor Manager::choiceTensor() const
{
    return impl_->exportTensor(ExportID::Choice, TensorElementType::Int8,
        {impl_->cfg.numWorlds, impl_->cfg.numPlayers, 1});
}

Tensor Manager::foulCallTensor() const
{
    return impl_->exportTensor(ExportID::CalledFoul, TensorElementType::Int8,
        {impl_->cfg.numWorlds, impl_->cfg.numPlayers, 1});
}
}
