#ifndef CONSTS_HPP
#define CONSTS_HPP

#include <cmath>

constexpr int COURT_WIDTH = 94.0;
constexpr int COURT_HEIGHT = 50.0;

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
constexpr double MAX_V_CHANGE = 40.0;

constexpr int ACTIVE_PLAYERS = 10;


constexpr int PLAYER_STARTING_WITH_BALL = 5;
constexpr int NOT_PREVIOUSLY_SHOT = -1;
constexpr int FIRST_TEAM2_PLAYER = 5;

constexpr double D_T = 0.1;

// constexpr char ASSET_PATH[] = "assets/";
// constexpr char CONFIG_FILE[] = "config/settings.cfg";

#endif // CONSTS_HPP
