#include "helpers.hpp"
#include <cstdlib> 

namespace madsimple {

CourtPos updateCourtPosition(const CourtPos &current_pos, const Action &action) {
    CourtPos new_player_pos = current_pos;

    // Update the player positions, by just adding 1 right now. Here is where we can add random movement
    new_player_pos.x += new_player_pos.v * cos(new_player_pos.th) * D_T;
    new_player_pos.y += new_player_pos.v * sin(new_player_pos.th) * D_T;
    new_player_pos.facing += new_player_pos.om * D_T;

    float dx = action.vdes * cos(action.thdes);
    float dy = action.vdes * sin(action.thdes);

    float ax = new_player_pos.v * cos(new_player_pos.th);
    float ay = new_player_pos.v * sin(new_player_pos.th);

    float lx = dx - ax;
    float ly = dy - ay;

    float dist = sqrt(lx * lx + ly * ly);

    if (dist <= MAX_V_CHANGE * D_T){ // always true for now
        new_player_pos.v = action.vdes;
        new_player_pos.th = action.thdes;
    } 
    
    // else {
    //     ax += ((MAX_V_CHANGE * D_T) / dist) * lx;
    //     ay += ((MAX_V_CHANGE * D_T) / dist) * ly;
    //     new_player_pos.v = sqrt(ax * ax + ay * ay);
    //     new_player_pos.th = atan2(ay, ax);
    // }

    new_player_pos.om = action.omdes;
    // replace court_pos with our new positions
    return new_player_pos;
}

void updateShotBallState(BallState &current_ball, const BallStatus &ball_status){
    std::mt19937 gen;
    std::uniform_real_distribution<> dis(15.0, 20.0);
    current_ball.v = (float)dis(gen);

    if (ball_status.heldBy > 4){ // inline helper functions
        current_ball.th = atan2(RIGHT_HOOP_Y - current_ball.y, RIGHT_HOOP_X - current_ball.x); // any number, make a const in constants.hpp
    } else {
        current_ball.th = atan2(LEFT_HOOP_Y - current_ball.y, LEFT_HOOP_X - current_ball.x);
    }
    
    current_ball.x += current_ball.v * cos(current_ball.th) * D_T;
    current_ball.y += current_ball.v * sin(current_ball.th) * D_T;
}

void changeBallToInPass(Engine &ctx, float th, float v, PlayerID &id) {
    BallStatus* status = &ctx.get<BallStatus>(ctx.singleton<BallReference>().theBall);
    BallState* state = &ctx.get<BallState>(ctx.singleton<BallReference>().theBall);
    status->ballState = BallStatesPossibilities::BALL_IN_PASS;
    status->heldBy = -1;
    status->whoPassed = id.id;
    state->th = th;
    state->v = v;
}

bool isHoldingBall(PlayerID &id, Engine &ctx) {
    return ctx.get<BallStatus>(ctx.singleton<BallReference>().theBall).heldBy == id.id;
} 

bool isBallLoose(Engine &ctx) {
    BallStatus* status = &ctx.get<BallStatus>(ctx.singleton<BallReference>().theBall);
    return status->heldBy == -1 && status->ballState == BallStatesPossibilities::BALL_IN_LOOSE;;
}

bool isBallInPass(Engine &ctx, PlayerID &id) {
    BallStatus* status = &ctx.get<BallStatus>(ctx.singleton<BallReference>().theBall);
    return (id.id != status->whoPassed) && (status->heldBy == -1) && (status->ballState == BallStatesPossibilities::BALL_IN_PASS);
}

bool canBallBeCaught(Engine &ctx, PlayerID &id) {
    return isBallInPass(ctx, id) || isBallLoose(ctx);
}

bool catchBallIfClose(Engine &ctx,
                      CourtPos &court_pos,
                      PlayerID &id, 
                      PlayerStatus &status) {
    BallState* state = &ctx.get<BallState>(ctx.singleton<BallReference>().theBall);
    int x = state->x;
    int y = state->y;

    int player_x = court_pos.x;
    int player_y = court_pos.y;

    BallStatus* ball_status = &ctx.get<BallStatus>(ctx.singleton<BallReference>().theBall);

    if (std::abs(player_x - x) < CATCHING_WINGSPAN && std::abs(player_y - y) < CATCHING_WINGSPAN) 
    {
        status.hasBall = true;

        ball_status->heldBy = id.id;
        state->v = 0;
        state->th = 0;
        return true;
    }
    return false;
}

bool ballIsHeld(BallStatus &ball_held) {
    return ball_held.heldBy != -1;
}

} 