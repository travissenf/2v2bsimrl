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
    GridState *gridData;

    // Added courtData structure, which contains number of players, and array of players and their locations
    CourtState *courtData;

    // Added court_state ot constructor, which gives input to courtData
    inline Impl(const Config &c,
                EpisodeManager *ep_mgr,
                GridState *grid_data,
                CourtState *court_state)
        : cfg(c),
          episodeMgr(ep_mgr),
          gridData(grid_data),
          courtData(court_state)
    {}

    inline virtual ~Impl() {}

    virtual void run() = 0;
    virtual Tensor exportTensor(ExportID slot, TensorElementType type,
                                Span<const int64_t> dims) = 0;

    // Add CourtState to constructor
    static inline Impl * init(const Config &cfg, const GridState &src_grid, const CourtState &src_players);
};

struct Manager::CPUImpl final : Manager::Impl {
    using ExecT = TaskGraphExecutor<Engine, Sim, Sim::Config, WorldInit>;
    ExecT cpuExec;

    // Add courtData to constructor
    inline CPUImpl(const Manager::Config &mgr_cfg,
                   const Sim::Config &sim_cfg,
                   EpisodeManager *episode_mgr,
                   GridState *grid_data,
                   CourtState *court_data,
                   WorldInit *world_inits)
        : Impl(mgr_cfg, episode_mgr, grid_data, court_data),
          cpuExec({
                  .numWorlds = mgr_cfg.numWorlds,
                  .numExportedBuffers = (uint32_t)ExportID::NumExports,
              }, sim_cfg, world_inits, 1)
    {}

    // Free courtData
    inline virtual ~CPUImpl() final {
        delete episodeMgr;
        free(gridData);
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
                   GridState *grid_data,
                   CourtState *court_data,
                   WorldInit *world_inits)
        : Impl(mgr_cfg, episode_mgr, grid_data, court_data),
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
        REQ_CUDA(cudaFree(gridData));
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
                                               const GridState *grid,
                                               const CourtState *court)
{
    HeapArray<WorldInit> world_inits(num_worlds);

    for (int64_t i = 0; i < num_worlds; i++) {
        world_inits[i] = WorldInit {
            episode_mgr,
            grid, 
            court,
        };
    }

    return world_inits;
}

// Added CourtState to this
Manager::Impl * Manager::Impl::init(const Config &cfg,
                                    const GridState &src_grid,
                                    const CourtState &src_court)
{
    static_assert(sizeof(GridState) % alignof(Cell) == 0);

    Sim::Config sim_cfg {
        .maxEpisodeLength = cfg.maxEpisodeLength,
        .enableViewer = false,
    };

    switch (cfg.execMode) {
    case ExecMode::CPU: {
        EpisodeManager *episode_mgr = new EpisodeManager { 0 };

        uint64_t num_cell_bytes =
            sizeof(Cell) * src_grid.width * src_grid.height;

        auto *grid_data =
            (char *)malloc(sizeof(GridState) + num_cell_bytes);
        Cell *cpu_cell_data = (Cell *)(grid_data + sizeof(GridState));

        GridState *cpu_grid = (GridState *)grid_data;
        *cpu_grid = GridState {
            .cells = cpu_cell_data,
            .startX = src_grid.startX,
            .startY = src_grid.startY,
            .width = src_grid.width,
            .height = src_grid.height,
        };

        memcpy(cpu_cell_data, src_grid.cells, num_cell_bytes);


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
            episode_mgr, cpu_grid, cpu_court);

        return new CPUImpl(cfg, sim_cfg, episode_mgr, cpu_grid, cpu_court, world_inits.data());
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

        uint64_t num_cell_bytes =
            sizeof(Cell) * src_grid.width * src_grid.height;

        auto *grid_data =
            (char *)cu::allocGPU(sizeof(GridState) + num_cell_bytes);

        Cell *gpu_cell_data = (Cell *)(grid_data + sizeof(GridState));
        GridState grid_staging {
            .cells = gpu_cell_data,
            .startX = src_grid.startX,
            .startY = src_grid.startY,
            .width = src_grid.width,
            .height = src_grid.height,
        };

        cudaMemcpy(grid_data, &grid_staging, sizeof(GridState),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_cell_data, src_grid.cells, num_cell_bytes,
                   cudaMemcpyHostToDevice);

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

        GridState *gpu_grid = (GridState *)grid_data;

        HeapArray<WorldInit> world_inits = setupWorldInitData(cfg.numWorlds,
            episode_mgr, gpu_grid, cpu_court);

        return new GPUImpl(cu_ctx, cfg, sim_cfg, episode_mgr, gpu_grid, cpu_court,
                           world_inits.data());
#endif
    } break;
    default: return nullptr;
    }
}

// Added initial conditions to manager
Manager::Manager(const Config &cfg,
                 const GridState &src_grid,
                 const CourtState &src_court)
    : impl_(Impl::init(cfg, src_grid, src_court))
{}

Manager::~Manager() {}

void Manager::step()
{
    impl_->run();
}

Tensor Manager::resetTensor() const
{
    return impl_->exportTensor(ExportID::Reset, TensorElementType::Int32,
                               {impl_->cfg.numWorlds, 1});
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

Tensor Manager::observationTensor() const
{
    return impl_->exportTensor(ExportID::GridPos, TensorElementType::Int32,
        {impl_->cfg.numWorlds, 2});
}

Tensor Manager::rewardTensor() const
{
    return impl_->exportTensor(ExportID::Reward, TensorElementType::Float32,
        {impl_->cfg.numWorlds, 1});
}

Tensor Manager::doneTensor() const
{
    return impl_->exportTensor(ExportID::Done, TensorElementType::Float32,
        {impl_->cfg.numWorlds, 1});
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

Tensor Manager::choiceTensor() const
{
    return impl_->exportTensor(ExportID::Choice, TensorElementType::Int8,
        {impl_->cfg.numWorlds, impl_->cfg.numPlayers, 1});
}
}
