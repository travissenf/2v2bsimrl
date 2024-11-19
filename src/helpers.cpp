#include "helpers.hpp"

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

    if (dist <= MAX_V_CHANGE * D_T){
        new_player_pos.v = action.vdes;
        new_player_pos.th = action.thdes;
    } else {
        ax += ((MAX_V_CHANGE * D_T) / dist) * lx;
        ay += ((MAX_V_CHANGE * D_T) / dist) * ly;
        new_player_pos.v = sqrt(ax * ax + ay * ay);
        new_player_pos.th = atan2(ay, ax);
    }
    new_player_pos.om = action.omdes;
    // replace court_pos with our new positions
    return new_player_pos;
}

BallState updateShotBallState(const BallState &current_ball, const BallStatus &ball_status){
    std::mt19937 gen;
    std::uniform_real_distribution<> dis(15.0, 20.0);
    BallState new_ball_state = current_ball;
    new_ball_state.v = (float)dis(gen);

    if (ball_status.heldBy > 4){ // inline helper functions
        new_ball_state.th = atan2(RIGHT_HOOP_Y - new_ball_state.y, RIGHT_HOOP_X - new_ball_state.x); // any number, make a const in constants.hpp
    } else {
        new_ball_state.th = atan2(LEFT_HOOP_Y - new_ball_state.y, LEFT_HOOP_X - new_ball_state.x);
    }
    
    new_ball_state.x += new_ball_state.v * cos(new_ball_state.th) * D_T;
    new_ball_state.y += new_ball_state.v * sin(new_ball_state.th) * D_T;
    return new_ball_state;
}



} 