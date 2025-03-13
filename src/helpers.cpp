#include "helpers.hpp"
#include <cstdlib> 
#include <cmath>
#include <algorithm>

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

float euclideanDistance(float x_1, float y_1, float x_2, float y_2) {
    return std::sqrt((x_2 - x_1) * (x_2 - x_1) + (y_2 - y_1) * (y_2 - y_1));
}

CourtPos updateCourtPositionStepped(const CourtPos &current_pos, const Action &action) {
    CourtPos new_player_pos = current_pos;
    float stepdt = D_T / COLLISION_CHECK_STEPS;

    float dx = action.vdes * cos(action.thdes);
    float dy = action.vdes * sin(action.thdes);

    float ax = new_player_pos.v * cos(new_player_pos.th);
    float ay = new_player_pos.v * sin(new_player_pos.th);

    float lx = dx - ax;
    float ly = dy - ay;

    float dist = sqrt(lx * lx + ly * ly);

    if (dist <= MAX_V_CHANGE * stepdt){ // always true for now
        new_player_pos.v = action.vdes;
        new_player_pos.th = action.thdes;
    } else { // Move partially along the direction
        float scale = (MAX_V_CHANGE * stepdt) / dist;
        float nx = ax + lx * scale;
        float ny = ay + ly * scale;
    
        new_player_pos.v = sqrt(nx * nx + ny * ny);
        new_player_pos.th = atan2(ny, nx);
    }

    new_player_pos.om = action.omdes;

    // Update the player positions, by just adding 1 right now. Here is where we can add random movement
    new_player_pos.x += new_player_pos.v * cos(new_player_pos.th) * stepdt;
    new_player_pos.y += new_player_pos.v * sin(new_player_pos.th) * stepdt;
    new_player_pos.facing += new_player_pos.om * stepdt;

    
    // replace court_pos with our new positions
    return new_player_pos;
}

CourtPos cancelPrevMovementStep(const CourtPos &current_pos, const Action &action) {
    CourtPos new_player_pos = current_pos;
    float stepdt = D_T / COLLISION_CHECK_STEPS;

    new_player_pos.x -= new_player_pos.v * cos(new_player_pos.th) * stepdt;
    new_player_pos.y -= new_player_pos.v * sin(new_player_pos.th) * stepdt;
    
    return new_player_pos;
}


bool ballIsOOB(BallState &ball_state) {
    if ((ball_state.x > MIN_X) && (ball_state.x < MAX_X) && 
        (ball_state.y > MIN_Y) && (ball_state.y < MAX_Y)){
        return false;
    }

    return true;
}

int findClosestInbound(BallState &ball_state){
    int closest = 0; // Start with the first point as the closest
    float minDistance = std::numeric_limits<float>::max();

    for (int i = 0; i < INBOUND_POINTS.size(); i++) {
        float distance = euclideanDistance(INBOUND_POINTS[i].x, INBOUND_POINTS[i].y, ball_state.x, ball_state.y);
        if (distance < minDistance) {
            minDistance = distance;
            closest = i;
        }
    }

    return closest;
}

bool isThreePointer(float x, float y, float hoopx){
    return ((euclideanDistance(x, y, hoopx, LEFT_HOOP_Y) > 23.75) //23'9'' away from hoop
        || (y > MAX_Y - 3)// top corner three
        || (y < MIN_Y + 3));// bottom corner three
}

int32_t updateShotBallState(BallState &current_ball, const BallStatus &ball_status){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(25.0, 45.0);
    current_ball.v = (float)dis(gen);

    const float HOOP_X = (ball_status.heldBy >= FIRST_TEAM2_PLAYER) ? RIGHT_HOOP_X : LEFT_HOOP_X;
    const float HOOP_Y = (ball_status.heldBy >= FIRST_TEAM2_PLAYER) ? RIGHT_HOOP_Y : LEFT_HOOP_Y;

    float prob = probabilityOfShot(euclideanDistance(current_ball.x, current_ball.y, HOOP_X, HOOP_Y)); 

    // Generate a random chance for the decision
    std::uniform_real_distribution<> chance_dis(0.0, 100.0);
    float random_chance = static_cast<float>(chance_dis(gen));

    

    // Calculate the base theta angle
    float base_th = atan2(HOOP_Y - current_ball.y, HOOP_X - current_ball.x);

    // Decide if the correct or perturbed angle should be assigned
    current_ball.th = base_th;
    
    current_ball.x += current_ball.v * cos(current_ball.th) * D_T;
    current_ball.y += current_ball.v * sin(current_ball.th) * D_T;

    if (random_chance > prob){
        return 0;
    }
    else if (isThreePointer(current_ball.x, current_ball.y, HOOP_X))
    {
        return 3;
    }
    
    return 2;
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
    BallStatus* ball_status = &ctx.get<BallStatus>(ctx.singleton<BallReference>().theBall);

    if (ball_status->ballState == BallStatesPossibilities::T1_NEED_TO_INBOUND){
        if (id.id / FIRST_TEAM2_PLAYER != 0){
            return false;
        }
    } else if (ball_status->ballState == BallStatesPossibilities::T2_NEED_TO_INBOUND){
        if (id.id / FIRST_TEAM2_PLAYER != 1){
            return false;
        }
    }


    if (shouldPlayerCatch(state, court_pos)) 
    {
        status.hasBall = true;
        ball_status->heldBy = id.id;
        ball_status->whoShot = -1;
        ball_status->whoPassed = -1;

        state->v = 0;
        state->th = 0;
        
        return true;
    }
    return false;
}

bool ballIsHeld(BallStatus &ball_held) {
    return ball_held.heldBy != -1;
}

// just doing with distance for right now
float probabilityOfShot(float distance_from_basket) 
{
    const float max_probability = 100.0f; // 100% chance at 0 distance
    const float min_probability = 1.0f;   // 1% chance at very large distances
    const float decay_factor = -0.07f;     // Controls how fast probability decays

    float probability = max_probability * std::exp(decay_factor * distance_from_basket);
    return std::min(max_probability, std::max(min_probability, probability));
}

void makePlayerInboundBall(BallState &ball_state,
                           BallStatus &ball_status,
                           CourtPos &inbounding_player_position,
                           PlayerDecision &inbounding_player_decision,
                           PlayerStatus &inbounding_player_status,
                           PlayerID &id, 
                           CourtPos &other_player_position,
                           bool inboundLeft) {
    // // step 1: place player outside the court
    // if (inboundLeft) {
    //     inbounding_player_position[0] = LEFT_INBOUND_X;
    //     inbounding_player_position[1] = LEFT_INBOUND_Y;
    // } else {
    //     inbounding_player_position[0] = RIGHT_INBOUND_X;
    //     inbounding_player_position[1] = RIGHT_INBOUND_Y;
    // }
    // inbounding_player_position[2] = 0.0;
    // inbounding_player_position[3] = 0.0;

    // // step 2: give player ball
    // inbounding_player_status.hasBall = true;

    // // step 3: put other player close to player outside of the court
    // if (inboundLeft) {
    //     other_player_position[0] = LEFT_INBOUND_X + 10;
    //     other_player_position[1] = LEFT_INBOUND_Y;
    // } else {
    //     other_player_position[0] = RIGHT_INBOUND_X - 10;
    //     other_player_position[1] = RIGHT_INBOUND_Y
    // }
    // other_player_position[2] = 0.0;
    // other_player_position[3] = 0.0; 

    // // step 4: make player with ball, pass ball 
    // inbounding_player_decision = PlayerDecision::PASS;

    // // step5: update ball status and location
    // ball_state[0] = inbounding_player_position[0];
    // ball_state[1] = inbounding_player_position[1];
    // ball_status.heldBy = id;
    return;
}
}