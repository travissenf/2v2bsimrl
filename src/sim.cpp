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
    registry.registerComponent<BallStatus>();
    registry.registerComponent<BallReference>();
    registry.registerComponent<PlayerID>();
    registry.registerComponent<AgentList>();
    registry.registerComponent<PlayerStatus>();

    registry.registerArchetype<BallArchetype>();
    registry.registerArchetype<Agent>();

    registry.registerSingleton<BallReference>();
    registry.registerSingleton<AgentList>();

    // registry.registerArchetype<PlayerAgent>();

    // Export tensors for pytorch
    registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
    registry.exportColumn<Agent, CourtPos>((uint32_t)ExportID::CourtPos);
    registry.exportColumn<Agent, PlayerStatus>((uint32_t)ExportID::Choice);

    registry.exportColumn<BallArchetype, BallState>((uint32_t)ExportID::BallLoc);
    registry.exportColumn<BallArchetype, BallStatus>((uint32_t)ExportID::WhoHolds);

}

inline void takePlayerAction(Engine &ctx,
                 Action &action,
                 CourtPos &court_pos,
                 PlayerID &id, 
                 PlayerStatus &status)
                //  
                
{
    action.vdes = std::min(action.vdes, (float)30.0);

    switch (status.playerDecision) {
        case PlayerDecision::SHOOT: {
            if (ctx.get<BallStatus>(ctx.singleton<BallReference>().theBall).heldBy == id.id){
                PlayerStatus s = status;
                s.hasBall = false;
                s.justShot = true;
                status = s;
            }
            break;
        } 
        case PlayerDecision::PASS: {

        }
        case PlayerDecision::MOVE: {
            court_pos = updateCourtPosition(court_pos, action);
        }

        default: break;
    }

    // bool episode_done = false;
    // if (reset.resetNow != 0) {
    //     reset.resetNow = 0;
    //     episode_done = true;
    // }


    
    // court_pos = updateCourtPosition(court_pos, action);

}

inline void balltick(Engine &ctx,
                BallState &ball_state,
                BallStatus &ball_held)
                //  
{
    float dt = ctx.data().dt;
    BallState new_ball_state = ball_state;
    BallStatus new_ball_held = ball_held;
    auto players = ctx.singleton<AgentList>().e;
    float hoopx = LEFT_HOOP_X;
    float hoopy = LEFT_HOOP_Y;
    std::mt19937 gen; // single funciton call getRandomNumber between 0 and 1 
    std::uniform_real_distribution<> dis(15.0, 20.0);

    if (ball_held.heldBy != -1){
        Entity p = players[ball_held.heldBy];
        if (ctx.get<PlayerStatus>(p).justShot){
            new_ball_state = updateShotBallState(new_ball_state, new_ball_held);
            new_ball_held.whoShot = ball_held.heldBy;
            new_ball_held.heldBy = -1;
        } else {

            // ball updates with held player
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
        Action, CourtPos, PlayerID, PlayerStatus>>({});
    auto ballfunc = builder.addToGraph<ParallelForNode<Engine, balltick,
        BallState, BallStatus>>({tickfunc}); 
    builder.addToGraph<ParallelForNode<Engine, postprocess,
        PlayerStatus>>({ballfunc});
}

Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr),
      court(init.court),
      dt(D_T),
      maxEpisodeLength(cfg.maxEpisodeLength)
{
    ctx.singleton<BallReference>().theBall = ctx.makeEntity<BallArchetype>();
    ctx.get<BallState>(ctx.singleton<BallReference>().theBall) = BallState {CENTER_X, CENTER_Y, CENTER_Z,};
    ctx.get<BallStatus>(ctx.singleton<BallReference>().theBall) = BallStatus {PLAYER_STARTING_WITH_BALL, NOT_PREVIOUSLY_SHOT};

    for (int i = 0; i < court->numPlayers; i++){
        Entity agent = ctx.makeEntity<Agent>();
        ctx.get<Action>(agent) = Action {
            CENTER_X, CENTER_Y, CENTER_Z,
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
