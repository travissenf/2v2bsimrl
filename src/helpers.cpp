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

void changeBallToInPass(Engine &ctx, float th, float v, PlayerStatus &player_status, PlayerID &id) {
    BallStatus* status = &ctx.get<BallStatus>(ctx.singleton<BallReference>().theBall);
    BallState* state = &ctx.get<BallState>(ctx.singleton<BallReference>().theBall);
    status->ballState = BallStatesPossibilities::BALL_IN_PASS;
    
    player_status.hasBall = false;

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

bool shouldPlayerCatch(BallState *state, CourtPos &court_pos) {
    float ball_x = state->x;
    float ball_y = state->y;

    float player_x = court_pos.x;
    float player_y = court_pos.y;

    // if within catching range 
    if (!(std::abs(player_x - ball_x) < CATCHING_WINGSPAN 
          && std::abs(player_y - ball_y) < CATCHING_WINGSPAN)) {
        return false;
    } 

    // calculate direction the pass is coming from
    float angle_of_pass = atan2(player_y - ball_y, player_x - ball_x);

    // if facing a reasonable angle to get the catch
    // assumption is that if your direction is less than 45 degree away from ball
    // than you can't catch it (as your back is facing the ball)
    if (std::abs(angle_of_pass - state->th) < RADIANS_OF_45_DEGREES) {
        return false;
    }

    float smaller_dt = D_T / 20;
    // idea here is we want a small enough dt such that we are basically taking 
    // a derivative. if we use dt normally we run the risk of the next_ball_pos 
    // being further when both are in the same direction

    float next_ball_pos_x = state->v * cos(state->th) * smaller_dt + ball_x;
    float next_ball_pos_y = state->v * sin(state->th) * smaller_dt + ball_y;

    float dist_curr = sqrt((ball_x - player_x) * (ball_x - player_x) +
                           (ball_y - player_y) * (ball_y - player_y));
    float next_distance = 
        sqrt((next_ball_pos_x - player_x) * (next_ball_pos_x - player_x) +
             (next_ball_pos_y - player_y) * (next_ball_pos_y - player_y));
    // We are doing these checks to make sure we aren't catching a ball that's 
    // moving away from a player (in the opposite direction)
    if (next_distance > dist_curr) {
        return false;
    }

    return true;
}

bool catchBallIfClose(Engine &ctx,
                      CourtPos &court_pos,
                      PlayerID &id, 
                      PlayerStatus &status) {
    BallState* state = &ctx.get<BallState>(ctx.singleton<BallReference>().theBall);
    float ball_x = state->x;
    float ball_y = state->y;

    float player_x = court_pos.x;
    float player_y = court_pos.y;

    if (shouldPlayerCatch(state, court_pos)) 
    {
        status.hasBall = true;

        BallStatus* ball_status = &ctx.get<BallStatus>(ctx.singleton<BallReference>().theBall);
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

// Ask roy for what a reasonable simple formula would be for the probability
// of a shot given distance, contension, and shot making percentages
float probabilityOfShot(float distance_from_basket,
                        float contension,
                        float shot_make_percentage) 
{
    float net_distance = distance_from_basket - 18;
    float net_contension = contension - 50;
    float new_percentage = shot_make_percentage + net_distance + net_contension;
    return std::min((float)99.0, std::max((float)1.0, new_percentage));
} 

}