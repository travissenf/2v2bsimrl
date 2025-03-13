#include "sim.hpp"
#include "helpers.hpp"
#include <madrona/mw_gpu_entry.hpp>
#include <random>
#include <cmath>
#include <iostream>


using namespace madrona;
using namespace madrona::math;

namespace madsimple {

void Sim::registerTypes(ECSRegistry &registry, const Config &)
{
    base::registerTypes(registry);

    registry.registerComponent<Action>();
    registry.registerComponent<CourtPos>();
    registry.registerComponent<StaticPlayerAttributes>();
    registry.registerComponent<BallState>();
    registry.registerComponent<BallStatus>();
    registry.registerComponent<BallReference>();
    registry.registerComponent<PlayerID>();
    registry.registerComponent<AgentList>();
    registry.registerComponent<PlayerStatus>();
    registry.registerComponent<PlayerDecision>();
    registry.registerComponent<FoulID>();
    registry.registerComponent<Scorecard>();

    registry.registerArchetype<BallArchetype>();
    registry.registerArchetype<Agent>();
    registry.registerArchetype<GameState>();

    registry.registerSingleton<BallReference>();
    registry.registerSingleton<AgentList>();
    registry.registerSingleton<GameReference>();
    registry.registerSingleton<WorldReset>();

    // registry.registerArchetype<PlayerAgent>();

    // Export tensors for pytorch
    registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
    registry.exportColumn<Agent, CourtPos>((uint32_t)ExportID::CourtPos);
    registry.exportColumn<Agent, PlayerDecision>((uint32_t)ExportID::Choice);
    registry.exportColumn<Agent, FoulID>((uint32_t)ExportID::CalledFoul);
    registry.exportColumn<Agent, StaticPlayerAttributes>((uint32_t)ExportID::StaticPlayerAttributes);

    registry.exportColumn<GameState, Scorecard>((uint32_t)ExportID::Scorecard);

    registry.exportColumn<BallArchetype, BallState>((uint32_t)ExportID::BallLoc);
    registry.exportColumn<BallArchetype, BallStatus>((uint32_t)ExportID::WhoHolds);

    registry.exportSingleton<WorldReset>((uint32_t)ExportID::Reset);

}

inline void takePlayerAction(Engine &ctx,
                Action &action,
                 CourtPos &court_pos,
                 PlayerID &id, 
                 PlayerStatus &status, 
                 PlayerDecision &decision, 
                 FoulID &foul)
                //  
                
{

    foul = FoulID::NO_CALL; // reset foul state
    if (canBallBeCaught(ctx, id)) {
        if (catchBallIfClose(ctx, court_pos, id, status)) {
            return;
        }
    }
    status.justShot = false;
    if (isHoldingBall(id, ctx)){
        status.hasBall = true;
    } else {
        status.hasBall = false;
    }
    switch (decision) {
        case PlayerDecision::SHOOT: {
            if (isHoldingBall(id, ctx)){
                status.hasBall = false;
                status.justShot = true;

                BallStatus* ball_status = &ctx.get<BallStatus>(ctx.singleton<BallReference>().theBall);
                ball_status->ballState = BallStatesPossibilities::BALL_IN_SHOT;
            }
            break;
        } 
        case PlayerDecision::PASS: {
            if (isHoldingBall(id, ctx)) {
                status.hasBall = false;
                status.justShot = false;

                changeBallToInPass(ctx, action.pass_th, action.pass_v, status, id);
            }
            break;
        }
        case PlayerDecision::MOVE: {
            // std::cout << "in TAKE PLAYER ACTION: decision is move" << std::endl;
            // if (!isHoldingBall(id, ctx) && isBallLoose()) {

            // }
        }
        case PlayerDecision::NOTHING: {

        }

        default: break;
    }
}

inline void movePlayerStep(Engine &ctx,
                     Action &action,
                     CourtPos &court_pos)
{
    action.vdes = std::min(action.vdes, (float)30.0);
    court_pos = updateCourtPositionStepped(court_pos, action);
}


inline void checkForBlockCharge(Engine &ctx,
                 Action &action,
                 CourtPos &court_pos,
                 PlayerID &id, 
                 PlayerStatus &status, 
                 PlayerDecision &decision,
                 FoulID &foul)
{
    auto players = ctx.singleton<AgentList>().e;
    for (int i = 0; i < ACTIVE_PLAYERS; i++){
        if (i == id.id){
            continue;
        }
        Entity p = players[i];

        float distance = std::sqrt(std::pow(
            court_pos.x - ctx.get<CourtPos>(p).x, 2) + std::pow(court_pos.y - ctx.get<CourtPos>(p).y, 2
            ));

         if (distance <= 1.5){ // If they collided, check
            if ((i / FIRST_TEAM2_PLAYER) == (id.id / FIRST_TEAM2_PLAYER)){ // if same team
                court_pos = cancelPrevMovementStep(court_pos, action); // revert the move
            } else {
                int whoHasBall = ctx.get<BallStatus>(ctx.singleton<BallReference>().theBall).heldBy;
                if (whoHasBall == -1){
                    whoHasBall = ctx.get<BallStatus>(ctx.singleton<BallReference>().theBall).whoPassed;
                }
                if (whoHasBall == -1){
                    whoHasBall = ctx.get<BallStatus>(ctx.singleton<BallReference>().theBall).whoShot;
                }

                float v1_x = -1 * court_pos.v * cos(court_pos.th);
                float v1_y = -1 * court_pos.v * sin(court_pos.th);
                float v2_x = ctx.get<CourtPos>(p).v * cos(ctx.get<CourtPos>(p).th);
                float v2_y = ctx.get<CourtPos>(p).v * sin(ctx.get<CourtPos>(p).th);
            
                
                float impact_factor = sqrt(pow(v1_x + v2_x, 2) + pow(v1_y + v2_y, 2));

                if ((ctx.get<CourtPos>(p).v < 0.5) && (court_pos.v < 0.5)){ // if both players arent really moving
                    // do nothing
                } else if (((id.id / FIRST_TEAM2_PLAYER) == (whoHasBall / FIRST_TEAM2_PLAYER))
                    && (id.id != whoHasBall)){ // If we are off ball on offense
                    if ((impact_factor >= 1.0) && (court_pos.v >= 0.5)){ // and we run into them
                        foul = FoulID::CHARGE;
                    }
                } else if (id.id == whoHasBall){ // If we have the ball
                    if (ctx.get<CourtPos>(p).v < 0.5){ // and they are not moving
                        foul = FoulID::CHARGE;
                    }
                } else if (i == whoHasBall) { // If on defense, and player we collide with has the ball
                    if ((impact_factor >= 1.0) && (court_pos.v >= 0.5)){ // if we are moving
                        foul = FoulID::BLOCK;
                    }
                } else if ((id.id / FIRST_TEAM2_PLAYER) != (whoHasBall / FIRST_TEAM2_PLAYER)
                    && (i != whoHasBall)){ // if on defense, player with we collide with doesnt have ball
                    if (ctx.get<CourtPos>(p).v < 0.5) { // if they are not moving
                        foul = FoulID::PUSH;
                    }
                }
                if (foul != FoulID::NO_CALL){
                    court_pos = cancelPrevMovementStep(court_pos, action); // revert the move
                }
            }
         } 
    }
}


inline void balltick(Engine &ctx,
                     BallState &ball_state,
                     BallStatus &ball_held)
                //  
{
    float dt = ctx.data().dt;
    auto players = ctx.singleton<AgentList>().e;
    std::random_device rd;
    std::mt19937 gen(rd()); // single funciton call getRandomNumber between 0 and 1 
    std::uniform_real_distribution<> dis(15.0, 20.0);

    if (ballIsHeld(ball_held)){
        Entity p = players[ball_held.heldBy];
        if (ctx.get<PlayerStatus>(p).justShot){
            ctx.get<PlayerStatus>(p).pointsOnMake = updateShotBallState(ball_state, ball_held);
            ball_held.whoShot = ball_held.heldBy;
            ball_held.heldBy = -1;
        } else {

            // ball updates with held player
            ball_state.x = ctx.get<CourtPos>(p).x;
            ball_state.y = ctx.get<CourtPos>(p).y;
            ball_state.v = ctx.get<CourtPos>(p).v;
            ball_state.th = ctx.get<CourtPos>(p).th;
        }

    } else {
        float hoopx = LEFT_HOOP_X;

        float old_ball_state_x = ball_state.x;
        float old_ball_state_y = ball_state.y;
        ball_state.x += ball_state.v * cos(ball_state.th) * dt;
        ball_state.y += ball_state.v * sin(ball_state.th) * dt;
        if (ball_held.whoShot > -1){
            bool team1 = true;
            if (ball_held.whoShot >= FIRST_TEAM2_PLAYER){
                hoopx = RIGHT_HOOP_X;
                team1 = false;
            }
            if (sqrt((hoopx - ball_state.x) * (hoopx - ball_state.x) + (LEFT_HOOP_Y - ball_state.y) * (LEFT_HOOP_Y - ball_state.y))
            <= sqrt((ball_state.x - old_ball_state_x) * (ball_state.x - old_ball_state_x) 
            + (ball_state.y - old_ball_state_y) * (ball_state.y - old_ball_state_y))) {
                // did shot go in?
                Entity p = players[ball_held.whoShot];
                if (ctx.get<PlayerStatus>(p).pointsOnMake != 0){
                    ball_state.v = 0.0;
                    ball_state.th = 0.0;
                    if (team1){
                        ctx.get<Scorecard>(ctx.singleton<GameReference>().theGame).score1 
                            += ctx.get<PlayerStatus>(p).pointsOnMake;
                    } else {
                        ctx.get<Scorecard>(ctx.singleton<GameReference>().theGame).score2 
                            += ctx.get<PlayerStatus>(p).pointsOnMake;
                    }
                    
                    ctx.get<PlayerStatus>(p).pointsOnMake = 0;
                }
                else{
                    dis = std::uniform_real_distribution<>(0.0, 10.0);
                    ball_state.v = (float)dis(gen);
                    dis = std::uniform_real_distribution<>(atan(1)*-2, atan(1)*2);
                    ball_state.th = (float)dis(gen);
                }
                if (!team1) {
                    ball_state.th += atan(1) * 4;
                }

                ctx.get<PlayerStatus>(p).justShot = false;
                ball_held.whoShot = -1;
                
                
            }
        } else {
            for (int i = 0; i < ACTIVE_PLAYERS; i++){
                Entity pl = players[i];
                CourtPos ppos = ctx.get<CourtPos>(pl);
                float dist = sqrt((ppos.x - ball_state.x) * (ppos.x - ball_state.x) + (ppos.y - ball_state.y) * (ppos.y - ball_state.y));
                if ((dist < 2.0) && (ctx.get<PlayerID>(pl).id != ball_held.whoPassed)) {
                    ball_held.heldBy = i;
                    ball_held.ballState = BallStatesPossibilities::BALL_IS_HELD;
                    ball_held.whoPassed = -1;
                    ball_held.whoShot = -1; // rebounding check here?
                    ball_state.x = ppos.x;
                    ball_state.y = ppos.y;
                    ball_state.v = ppos.v;
                    ball_state.th = ppos.th;
                    break;
                }
            }
        }
    }

    // if (ballIsOOB(ball_state) && (ball_held.ballState < 3)){
    //     int closest_inbound_index = findClosestInbound(ball_state);
    //     ball_state.x = INBOUND_POINTS[closest_inbound_index].x;
    //     ball_state.y = INBOUND_POINTS[closest_inbound_index].y;
    //     ball_state.v = 0.0;


    //     // determine who needs to inbound the ball
    //     if (ball_held.heldBy >= 0){
    //         if (ball_held.heldBy < FIRST_TEAM2_PLAYER){
    //             ball_held.ballState = BallStatesPossibilities::T2_NEED_TO_INBOUND;
    //         } else {
    //             ball_held.ballState = BallStatesPossibilities::T1_NEED_TO_INBOUND;
    //         }
    //     } else if (ball_held.whoPassed >= 0){
    //         if (ball_held.whoPassed < FIRST_TEAM2_PLAYER){
    //             ball_held.ballState = BallStatesPossibilities::T2_NEED_TO_INBOUND;
    //         } else {
    //             ball_held.ballState = BallStatesPossibilities::T1_NEED_TO_INBOUND;
    //         }
    //     } else if (ball_held.whoShot >= 0 ){
    //         if (ball_held.whoShot < FIRST_TEAM2_PLAYER){
    //             ball_held.ballState = BallStatesPossibilities::T2_NEED_TO_INBOUND;
    //         } else {
    //             ball_held.ballState = BallStatesPossibilities::T1_NEED_TO_INBOUND;
    //         }
    //     }
    // } // saving inbound code for later


    ctx.get<Scorecard>(ctx.singleton<GameReference>().theGame).ticksElapsed += 1;
}

inline void postprocess(Engine &ctx,
                PlayerID &id,
                PlayerStatus &status)
                //  
{
    PlayerStatus st = status;
    if (ctx.get<BallStatus>(ctx.singleton<BallReference>().theBall).heldBy != id.id) {
        st.hasBall = false;
    }
    st.justShot = false;
    status = st;
}

void Sim::setupTasks(TaskGraphManager &taskgraph_mgr,
                     const Config &)
{
    TaskGraphBuilder &builder = taskgraph_mgr.init(0);
    
    auto actionfunc = builder.addToGraph<ParallelForNode<Engine, takePlayerAction,
        Action, CourtPos, PlayerID, PlayerStatus, PlayerDecision, FoulID>>({});

    auto movementfunc = builder.addToGraph<ParallelForNode<Engine, movePlayerStep,
        Action, CourtPos>>({actionfunc});

    auto blockchargecheck = builder.addToGraph<ParallelForNode<Engine, checkForBlockCharge,
        Action, CourtPos, PlayerID, PlayerStatus, PlayerDecision, FoulID>>({movementfunc});

    for (int i = 1; i < COLLISION_CHECK_STEPS; i++){

        movementfunc = builder.addToGraph<ParallelForNode<Engine, movePlayerStep,
            Action, CourtPos>>({blockchargecheck});

        blockchargecheck = builder.addToGraph<ParallelForNode<Engine, checkForBlockCharge,
            Action, CourtPos, PlayerID, PlayerStatus, PlayerDecision, FoulID>>({movementfunc});
    }

    auto ballfunc = builder.addToGraph<ParallelForNode<Engine, balltick,
        BallState, BallStatus>>({blockchargecheck});

    builder.addToGraph<ParallelForNode<Engine, postprocess, PlayerID,
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
    if (PLAYER_STARTING_WITH_BALL != -1) {
        ctx.get<BallStatus>(ctx.singleton<BallReference>().theBall) = BallStatus {PLAYER_STARTING_WITH_BALL, NOT_PREVIOUSLY_SHOT, -1, BallStatesPossibilities::BALL_IS_HELD};
    } else {
        ctx.get<BallStatus>(ctx.singleton<BallReference>().theBall) = BallStatus {PLAYER_STARTING_WITH_BALL, NOT_PREVIOUSLY_SHOT, -1, BallStatesPossibilities::BALL_IN_LOOSE};
    }
    ctx.singleton<GameReference>().theGame = ctx.makeEntity<GameState>();
    ctx.get<Scorecard>(ctx.singleton<GameReference>().theGame) = Scorecard {0, 0, 1, 0};


    for (int i = 0; i < ACTIVE_PLAYERS; i++){
        Entity agent = ctx.makeEntity<Agent>();
        ctx.get<Action>(agent) = Action {
            CENTER_X, CENTER_Y, CENTER_Z, 0.0, 0.0
        };
        ctx.get<CourtPos>(agent) = CourtPos {
            court->players[i].x, court->players[i].y, 
            court->players[i].th, court->players[i].v, 
            court->players[i].om, court->players[i].facing,
        };
        ctx.get<PlayerID>(agent).id = i;
        ctx.get<PlayerStatus>(agent) = {false, false, 0};
        ctx.get<FoulID>(agent) = FoulID::NO_CALL;
        ctx.get<StaticPlayerAttributes>(agent) = {0.0, 0.0, 0.0};
        ctx.singleton<AgentList>().e[i] = agent;
    }
    
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
