#pragma once
#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <cmath>
#include <random>

#include "sim.hpp"

namespace madsimple {

// Function declarations
CourtPos updateCourtPosition(const CourtPos &current_pos, const Action &action);

BallState updateBallState(const BallState &current_ball, const BallStatesPossibilities &ball_held, 
                          const madrona::Entity *players, const Engine &ctx, float dt);

BallState updateShotBallState(const BallState &current_ball, const BallStatus &ball_status);

float calculateDistance(float x1, float y1, float x2, float y2);

float generateRandomValue(float min_val, float max_val);

void resetBallState(BallState &ball_state, BallStatus &ball_status, float hoop_th);

} // namespace madsimple

#endif // HELPERS_HPP
