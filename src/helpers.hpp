#pragma once
#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <cmath>
#include <random>

#include "sim.hpp"

namespace madsimple {

// Function declarations
CourtPos updateCourtPosition(const CourtPos &current_pos, const Action &action);
CourtPos updateCourtPositionStepped(const CourtPos &current_pos, const Action &action);
CourtPos cancelPrevMovementStep(const CourtPos &current_pos, const Action &action);

BallState updateBallState(const BallState &current_ball, const BallStatesPossibilities &ball_held, 
                          const madrona::Entity *players, const Engine &ctx, float dt);

bool ballIsOOB(BallState &ball_state);
int findClosestInbound(BallState &ball_state);

bool isThreePointer(float x, float y, float hoopx);
int8_t updateShotBallState(BallState &current_ball, const BallStatus &ball_status);

float calculateDistance(float x1, float y1, float x2, float y2);

float generateRandomValue(float min_val, float max_val);

void resetBallState(BallState &ball_state, BallStatus &ball_status, float hoop_th);

bool isHoldingBall(PlayerID &id, Engine &ctx);
bool isBallLoose(Engine &ctx);
bool isBallInPass(Engine &ctx, PlayerID &id);

bool canBallBeCaught(Engine &ctx, PlayerID &id);
bool shouldPlayerCatch(BallState *state, CourtPos &court_pos);

bool ballIsHeld(BallStatus &ball_held);

void changeBallToInPass(Engine &ctx, 
                        float th, 
                        float v, 
                        PlayerStatus &player_status, 
                        PlayerID &id);

bool catchBallIfClose(Engine &ctx,
                      CourtPos &court_pos,
                      PlayerID &id, 
                      PlayerStatus &status);

float probabilityOfShot(float distance_from_basket);

void makePlayerInboundBall(BallState &ball_state,
                           BallStatus &ball_status,
                           CourtPos &inbounding_player_position,
                           PlayerDecision &inbounding_player_decision,
                           PlayerStatus &inbounding_player_status,
                           PlayerID &id, 
                           CourtPos &other_player_position,
                           bool inboundLeft);
} // namespace madsimple

#endif // HELPERS_HPP
