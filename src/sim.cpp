#include "sim.hpp"
#include "helpers.hpp"
#include <madrona/mw_gpu_entry.hpp>
#include <random>
#include <cmath>


using namespace madrona;
using namespace madrona::math;

namespace madsimple {

void Sim::registerTypes(ECSRegistry &registry, const Config &)
{
    base::registerTypes(registry);

    registry.registerComponent<Action>();
    registry.registerComponent<CourtPos>();
    registry.registerComponent<BallState>();
    registry.registerComponent<BallHeld>();
    registry.registerComponent<BallReference>();
    registry.registerComponent<PlayerID>();
    registry.registerComponent<AgentList>();
    registry.registerComponent<Decision>();
    registry.registerComponent<PlayerStatus>();

    registry.registerArchetype<BallArchetype>();
    registry.registerArchetype<Agent>();

    registry.registerSingleton<BallReference>();
    registry.registerSingleton<AgentList>();

    // registry.registerArchetype<PlayerAgent>();

    // Export tensors for pytorch
    registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
    registry.exportColumn<Agent, CourtPos>((uint32_t)ExportID::CourtPos);
    registry.exportColumn<Agent, Decision>((uint32_t)ExportID::Choice);

    registry.exportColumn<BallArchetype, BallState>((uint32_t)ExportID::BallLoc);
    registry.exportColumn<BallArchetype, BallHeld>((uint32_t)ExportID::WhoHolds);

}

inline void takePlayerAction(Engine &ctx,
                 Action &action,
                 CourtPos &court_pos,
                 Decision &decision,
                 PlayerID &id, 
                 PlayerStatus &status)
                //  
                
{
    switch (decision) {
        case Decision::Shoot: {
            if (ctx.get<BallHeld>(ctx.singleton<BallReference>().theBall).held == id.id){
                PlayerStatus s = status;
                s.hasBall = false;
                s.justShot = true;
                status = s;
            }
        } break;
        default: break;
    }

    // bool episode_done = false;
    // if (reset.resetNow != 0) {
    //     reset.resetNow = 0;
    //     episode_done = true;
    // }


    action.vdes = std::min(action.vdes, (float)30.0);
    court_pos = updateCourtPosition(court_pos, action);

}

inline void balltick(Engine &ctx,
                BallState &ball_state,
                BallHeld &ball_held)
                //  
{
    float dt = ctx.data().dt;
    BallState new_ball_state = ball_state;
    BallHeld new_ball_held = ball_held;
    auto players = ctx.singleton<AgentList>().e;
    float hoopx = LEFT_HOOP_X;
    float hoopy = LEFT_HOOP_Y;
    std::mt19937 gen; // single funciton call getRandomNumber between 0 and 1 
    std::uniform_real_distribution<> dis(15.0, 20.0);

    if (ball_held.held != -1){
        Entity p = players[ball_held.held];
        if (ctx.get<PlayerStatus>(p).justShot){
            new_ball_state.v = (float)dis(gen);
            if (ball_held.held > 4){ // inline helper functions
                hoopx = RIGHT_HOOP_X; // any number, make a const in constants.hpp
            }
            new_ball_state.th = atan2(hoopy - new_ball_state.y, hoopx - new_ball_state.x);
            new_ball_state.x += new_ball_state.v * cos(new_ball_state.th) * dt;
            new_ball_state.y += new_ball_state.v * sin(new_ball_state.th) * dt;
            new_ball_held.whoShot = ball_held.held;
            new_ball_held.held = -1;
        } else {
            new_ball_state.x = ctx.get<CourtPos>(p).x;
            new_ball_state.y = ctx.get<CourtPos>(p).y;
            new_ball_state.v = ctx.get<CourtPos>(p).v;
            new_ball_state.th = ctx.get<CourtPos>(p).th;
        }

    } else {
        new_ball_state.x += new_ball_state.v * cos(new_ball_state.th) * dt;
        new_ball_state.y += new_ball_state.v * sin(new_ball_state.th) * dt;
        if (ball_held.whoShot > -1){
            if (ball_held.whoShot > 4){
                hoopx = LEFT_HOOP_X;
            }
            if (sqrt((hoopx - ball_state.x) * (hoopx - ball_state.x) + (hoopy - ball_state.y) * (hoopy - ball_state.y))
            <= sqrt((new_ball_state.x - ball_state.x) * (new_ball_state.x - ball_state.x) 
            + (new_ball_state.y - ball_state.y) * (new_ball_state.y - ball_state.y))) {
                // did shot go in?
                dis = std::uniform_real_distribution<>(0.0, 10.0);
                new_ball_state.v = (float)dis(gen);
                new_ball_held.whoShot = -1;
                dis = std::uniform_real_distribution<>(atan(1)*-2, atan(1)*2);
                new_ball_state.th = (float)dis(gen);
                if (ball_held.whoShot > 4) {
                    new_ball_state.th += atan(1) * 4;
                }
            }
        }
    }

    ball_state = new_ball_state;
    ball_held = new_ball_held;
}

inline void postprocess(Engine &ctx,
                PlayerStatus &status)
                //  
{
    PlayerStatus st = status;
    st.justShot = false;
    status = st;
}

void Sim::setupTasks(TaskGraphManager &taskgraph_mgr,
                     const Config &)
{
    TaskGraphBuilder &builder = taskgraph_mgr.init(0);
    auto tickfunc = builder.addToGraph<ParallelForNode<Engine, takePlayerAction,
        Action, CourtPos, Decision, PlayerID,
        PlayerStatus>>({});
    auto ballfunc = builder.addToGraph<ParallelForNode<Engine, balltick,
        BallState, BallHeld>>({tickfunc}); 
    builder.addToGraph<ParallelForNode<Engine, postprocess,
        PlayerStatus>>({ballfunc});
}

Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr),
      court(init.court),
      dt(0.1),
      maxEpisodeLength(cfg.maxEpisodeLength)
{
    ctx.singleton<BallReference>().theBall = ctx.makeEntity<BallArchetype>();
    ctx.get<BallState>(ctx.singleton<BallReference>().theBall) = BallState {0.0, 0.0, 0.0, 0.0};
    ctx.get<BallHeld>(ctx.singleton<BallReference>().theBall) = BallHeld {5, -1};

    for (int i = 0; i < court->numPlayers; i++){
        Entity agent = ctx.makeEntity<Agent>();
        ctx.get<Action>(agent) = Action {
            0.0, 0.0, 0.0,
        };
        ctx.get<CourtPos>(agent) = CourtPos {
            court->players[i].x, court->players[i].y, 
            court->players[i].th, court->players[i].v, 
            court->players[i].om, court->players[i].facing,
        };
        ctx.get<PlayerID>(agent).id = i;
        ctx.get<PlayerStatus>(agent) = {false, false};
        ctx.singleton<AgentList>().e[i] = agent;
    }
    
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
