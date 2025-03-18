#ifndef CONSTS_HPP
#define CONSTS_HPP

#include <cmath>
#include <array>

struct Point2D {
    double x;
    double y;
};

constexpr std::array<Point2D, 6> INBOUND_POINTS = {{
    {18.0, -25.0},
    {18.0, 25.0},
    {-18.0, -25.0},
    {-18.0, 25.0},
    {47.0, 7.0},
    {-47.0, -7.0}
}};

constexpr float COURT_WIDTH = 94.0;
constexpr float COURT_HEIGHT = 50.0;

constexpr float MIN_X = -47.0;
constexpr float MAX_X = 47.0;
constexpr float MIN_Y = -25.0;
constexpr float MAX_Y = 25.0;

constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 2 * PI;
constexpr double HALF_PI = PI / 2;

constexpr double LEFT_HOOP_X = -41.75;
constexpr double LEFT_HOOP_Y = 0;
constexpr double RIGHT_HOOP_X = 41.75;
constexpr double RIGHT_HOOP_Y = 0;

constexpr double CENTER_X = 0.0;
constexpr double CENTER_Y = 0.0;
constexpr double CENTER_Z = 0.0;

constexpr double GRAVITY = 32.1741;
constexpr double MAX_V_CHANGE = 50.0;

constexpr int ACTIVE_PLAYERS = 4;
constexpr int COLLISION_CHECK_STEPS = 4;


constexpr int PLAYER_STARTING_WITH_BALL = 2;
constexpr int NOT_PREVIOUSLY_SHOT = -1;
constexpr int FIRST_TEAM2_PLAYER = ACTIVE_PLAYERS / 2;

constexpr double D_T = 0.05; //1;
constexpr double DECAY_FACTOR = 0.025;

constexpr double CATCHING_WINGSPAN = 2.75;

// used for calculating if pass/loose ball can be caught
// assumption is that if your direction is less than 45 degree away from ball
// than you can't catch it (as your back is facing the ball)
constexpr double RADIANS_OF_45_DEGREES = 0.78539;
// 180 degrees = pi => 45 degrees = pi / 180 * 45 

constexpr float LEFT_INBOUND_X = -40.0;
constexpr float RIGHT_INBOUND_X = 40.0;
constexpr float LEFT_INBOUND_Y = 10;
constexpr float RIGHT_INBOUND_Y = -10;

// constexpr char ASSET_PATH[] = "assets/";
// constexpr char CONFIG_FILE[] = "config/settings.cfg";

#endif // CONSTS_HPP
