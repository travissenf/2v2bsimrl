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

    if (dist <= 20.0 * D_T){
        new_player_pos.v = action.vdes;
        new_player_pos.th = action.thdes;
    } else {
        ax += ((20.0 * D_T) / dist) * lx;
        ay += ((20.0 * D_T) / dist) * ly;
        new_player_pos.v = sqrt(ax * ax + ay * ay);
        new_player_pos.th = atan2(ay, ax);
    }
    new_player_pos.om = action.omdes;
    // replace court_pos with our new positions
    return new_player_pos;
}

// BallState updateBallState(const BallState &current_ball, const BallHeld &ball_held, 
//                           const madrona::Entity *players, const madrona::Engine &ctx, float dt) {
//     // Implementation (same as suggested above)
//     ...
// }

// float calculateDistance(float x1, float y1, float x2, float y2) {
//     return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
// }

// float generateRandomValue(float min_val, float max_val) {
//     static std::mt19937 gen(std::random_device{}());
//     std::uniform_real_distribution<> dist(min_val, max_val);
//     return (float)dist(gen);
// }

// void resetBallState(BallState &ball_state, BallHeld &ball_held, float hoop_th) {
//     ball_state.v = generateRandomValue(0.0f, 10.0f);
//     ball_state.th = hoop_th + generateRandomValue(-M_PI / 2, M_PI / 2);
//     ball_held.whoShot = -1;
// }

} 