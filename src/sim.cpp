#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

void Sim::registerTypes(ECSRegistry &registry, const Config &)
{
    base::registerTypes(registry);

    registry.registerComponent<Reset>();
    registry.registerComponent<Action>();
    registry.registerComponent<GridPos>();
    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();
    registry.registerComponent<CurStep>();
    registry.registerComponent<CourtPos>();
    registry.registerComponent<BallState>();
    registry.registerComponent<BallHeld>();
    registry.registerComponent<BallReference>();
    registry.registerComponent<PlayerID>();

    registry.registerArchetype<BallArchetype>();
    registry.registerArchetype<Agent>();

    registry.registerSingleton<BallReference>();

    // registry.registerArchetype<PlayerAgent>();

    // Export tensors for pytorch
    registry.exportColumn<Agent, Reset>((uint32_t)ExportID::Reset);
    registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
    registry.exportColumn<Agent, GridPos>((uint32_t)ExportID::GridPos);
    registry.exportColumn<Agent, CourtPos>((uint32_t)ExportID::CourtPos);
    registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>((uint32_t)ExportID::Done);

    registry.exportColumn<BallArchetype, BallState>((uint32_t)ExportID::BallLoc);
    registry.exportColumn<BallArchetype, BallHeld>((uint32_t)ExportID::WhoHolds);

}

inline void tick(Engine &ctx,
                 Action &action,
                 Reset &reset,
                 GridPos &grid_pos,
                 Reward &reward,
                 Done &done,
                 CurStep &episode_step,
                 CourtPos &court_pos)
                //  
                
{
    const GridState *grid = ctx.data().grid;

    // This hopefully gets the player data, not entirely sure what how to use it
    // This function is next to be worked on
    // Idea: use input actions (which is randomized) to update player positions instead of just randomly generating?
    // const CourtState *court = ctx.data().court;

    GridPos new_pos = grid_pos;

    float dt = ctx.data().dt;
    //Old action code
    // switch (action) {
    //     case Action::Up: {
    //         new_pos.y += 1;
    //     } break;
    //     case Action::Down: {
    //         new_pos.y -= 1;
    //     } break;
    //     case Action::Left: {
    //         new_pos.x -= 1;
    //     } break;
    //     case Action::Right: {
    //         new_pos.x += 1;
    //     } break;
    //     default: break;
    // }


    // action = Action::None;

    if (new_pos.x < 0) {
        new_pos.x = 0;
    }

    if (new_pos.x >= grid->width) {
        new_pos.x = grid->width - 1;
    }

    if (new_pos.y < 0) {
        new_pos.y = 0;
    }

    if (new_pos.y >= grid->height) {
        new_pos.y = grid->height -1;
    }


    {
        const Cell &new_cell = grid->cells[new_pos.y * grid->width + new_pos.x];

        if ((new_cell.flags & CellFlag::Wall)) {
            new_pos = grid_pos;
        }
    }

    const Cell &cur_cell = grid->cells[new_pos.y * grid->width + new_pos.x];

    bool episode_done = false;
    if (reset.resetNow != 0) {
        reset.resetNow = 0;
        episode_done = true;
    }

    if ((cur_cell.flags & CellFlag::End)) {
        episode_done = true;
    }

    uint32_t cur_step = episode_step.step;

    if (cur_step == ctx.data().maxEpisodeLength - 1) {
        episode_done = true;
    }

    if (episode_done) {
        done.episodeDone = 1.f;

        new_pos = GridPos {
            grid->startY,
            grid->startX,
        };

        episode_step.step = 0;
    } else {
        done.episodeDone = 0.f;
        episode_step.step = cur_step + 1;
    }

    // Commit new position
    grid_pos = new_pos;
    reward.r = cur_cell.reward;

    // We cannot update court_pos directly, so we make a copy, update the copy, and then replace court_pos
    CourtPos new_player_pos = court_pos;

    // Update the player positions, by just adding 1 right now. Here is where we can add random movement
    new_player_pos.x += new_player_pos.v * cos(new_player_pos.th) * dt;
    new_player_pos.y += new_player_pos.v * sin(new_player_pos.th) * dt;
    new_player_pos.facing += new_player_pos.om * dt;
    new_player_pos.om += action.alpha * dt;
    float vxnew = new_player_pos.v * cos(new_player_pos.th) + action.accel * cos(action.th) * dt;
    float vynew = new_player_pos.v * sin(new_player_pos.th) + action.accel * sin(action.th) * dt;

    new_player_pos.v = std::sqrt(vxnew * vxnew + vynew * vynew);
    new_player_pos.th = std::atan2(vynew, vxnew);
    // replace court_pos with our new positions
    court_pos = new_player_pos;
}

void Sim::setupTasks(TaskGraphManager &taskgraph_mgr,
                     const Config &)
{
    TaskGraphBuilder &builder = taskgraph_mgr.init(0);
    builder.addToGraph<ParallelForNode<Engine, tick,
        Action, Reset, GridPos, Reward, Done, CurStep, CourtPos>>({});
}

Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr),
      grid(init.grid),
      court(init.court),
      dt(0.1),
      maxEpisodeLength(cfg.maxEpisodeLength)
{
    ctx.singleton<BallReference>().theBall = ctx.makeEntity<BallArchetype>();
    ctx.get<BallState>(ctx.singleton<BallReference>().theBall) = BallState {0.0, 0.0, 0.0, 0.0};
    ctx.get<BallHeld>(ctx.singleton<BallReference>().theBall) = BallHeld {-1};
    ctx.get<CurStep>(ctx.singleton<BallReference>().theBall).step = 0.0;

    for (int i = 0; i < court->numPlayers; i++){
        Entity agent = ctx.makeEntity<Agent>();
        ctx.get<Action>(agent) = Action {
            0.0, 0.0, 0.0,
        };
        ctx.get<GridPos>(agent) = GridPos {
            grid->startY,
            grid->startX,
        };
        ctx.get<CourtPos>(agent) = CourtPos {
            court->players[i].x, court->players[i].y, 
            court->players[i].th, court->players[i].v, 
            court->players[i].om, court->players[i].facing,
        };
        ctx.get<Reward>(agent).r = 0.f;
        ctx.get<Done>(agent).episodeDone = 0.f;
        ctx.get<CurStep>(agent).step = 0;
        ctx.get<PlayerID>(agent).id = i;
    }
    
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
