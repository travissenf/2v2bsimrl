#include "mgr.hpp"

#include <madrona/macros.hpp>
#include <madrona/py/bindings.hpp>

namespace madsimple {


// Example function for the grid example, initializing rewards grid (from prev repo)
static void setRewards(Cell *cells,
                       const float *rewards,
                       int64_t grid_x,
                       int64_t grid_y)
{
    for (int64_t y = 0; y < grid_y; y++) {
        for (int64_t x = 0; x < grid_x; x++) {
            int64_t idx = y * grid_x + x;
            cells[idx].reward = rewards[idx];
        }
    }
}
// Example function for the grid example, initializing walls (from prev repo)
static void tagWalls(Cell *cells,
                     const bool *walls,
                     int64_t grid_x,
                     int64_t grid_y)
{
    for (int64_t y = 0; y < grid_y; y++) {
        for (int64_t x = 0; x < grid_x; x++) {
            int64_t idx = y * grid_x + x;

            if (walls[idx]) {
                cells[idx].flags |= CellFlag::Wall;
            }
        }
    }
}
// Example function for the grid example, initializing end of grid (from prev repo)
static void tagEnd(Cell *cells,
                   const int32_t *end_cells,
                   int64_t num_end_cells,
                   int64_t grid_x,
                   int64_t grid_y)
{
    for (int64_t c = 0; c < num_end_cells; c++) {
        int64_t idx = c * 2;
        int64_t y = (int32_t)end_cells[idx];
        int64_t x = (int32_t)end_cells[idx + 1];

        if (x >= grid_x || y >= grid_y) {
            throw std::runtime_error("Out of range end cells");
        }

        cells[y * grid_x + x].flags |= CellFlag::End;
    }
}


// Example function for the grid example, initializing entire grid and calling helpers
static Cell * setupCellData(
    const nb::ndarray<bool, nb::shape<-1, -1>,
        nb::c_contig, nb::device::cpu> &walls,
    const nb::ndarray<float, nb::shape<-1, -1>,
        nb::c_contig, nb::device::cpu> &rewards,
    const nb::ndarray<int32_t, nb::shape<-1, 2>,
        nb::c_contig, nb::device::cpu> &end_cells,
    int64_t grid_x,
    int64_t grid_y)

{
    Cell *cells = new Cell[grid_x * grid_y]();

    setRewards(cells, rewards.data(), grid_x, grid_y);
    tagWalls(cells, walls.data(), grid_x, grid_y);
    tagEnd(cells, end_cells.data(),
        (int64_t)end_cells.shape(0), grid_x, grid_y);
    
    return cells;
}


// New function, takes in player objects by reference, and updates players with given positions, and assigns them an index
static void setPositions(Player *players, const float *positions, int64_t num_players){
    for(int64_t c = 0; c < num_players; c++){
        int64_t idx = c * 6;
        players[c].id = c;
        players[c].x = positions[idx];
        players[c].y = positions[idx + 1];
        players[c].th = positions[idx + 2];
        players[c].v = positions[idx + 3];
        players[c].om = positions[idx + 4];
        players[c].facing = positions[idx + 5];
    }
}

// Wrapper function to call helpers, and return our Player array object
static Player * setupPlayerData(
    const nb::ndarray<float, nb::shape<-1, 6>,
        nb::c_contig, nb::device::cpu> &init_player_pos,
    int64_t num_players)

{
    Player *players = new Player[num_players]();
    setPositions(players, init_player_pos.data(), num_players);
    return players;
}


NB_MODULE(_madrona_simple_example_cpp, m) {
    madrona::py::setupMadronaSubmodule(m);
    
    // Our world simulator object
    nb::class_<Manager> (m, "SimpleGridworldSimulator")
        .def("__init__", [](Manager *self,
                            nb::ndarray<bool, nb::shape<-1, -1>,
                                nb::c_contig, nb::device::cpu> walls,
                            nb::ndarray<float, nb::shape<-1, -1>,
                                nb::c_contig, nb::device::cpu> rewards,
                            nb::ndarray<int32_t, nb::shape<-1, 2>,
                                nb::c_contig, nb::device::cpu> end_cells,
                            nb::ndarray<float, nb::shape<-1, 6>,
                                nb::c_contig, nb::device::cpu> init_player_pos, // new input array for initial player pos
                            int64_t start_x,
                            int64_t start_y,
                            int64_t max_episode_length,
                            madrona::py::PyExecMode exec_mode,
                            int64_t num_worlds,
                            int64_t num_players, // given number of players (need to decide if we include all players or just playing players)
                            int64_t gpu_id) {
            int64_t grid_y = (int64_t)walls.shape(0);
            int64_t grid_x = (int64_t)walls.shape(1);

            if ((int64_t)rewards.shape(0) != grid_y ||
                (int64_t)rewards.shape(1) != grid_x) {
                throw std::runtime_error("walls and rewards shapes don't match");
            }

            Cell *cells =
                setupCellData(walls, rewards, end_cells, grid_x, grid_y);
            
            Player *players = setupPlayerData(init_player_pos, num_players); // call our player data setup function

            new (self) Manager(Manager::Config {
                .maxEpisodeLength = (uint32_t)max_episode_length,
                .execMode = exec_mode,
                .numWorlds = (uint32_t)num_worlds,
                .numPlayers = (uint32_t)num_players, // new, passing in num_players to config
                .gpuID = (int)gpu_id,
            }, GridState {
                .cells = cells,
                .startX = (int32_t)start_x,
                .startY = (int32_t)start_y,
                .width = (int32_t)grid_x,
                .height = (int32_t)grid_y,
            }, CourtState { // new, passing in our court state to the manager
                .players = players,
                .numPlayers = (int32_t)num_players
            });

            delete[] cells;
        }, nb::arg("walls"),
           nb::arg("rewards"),
           nb::arg("end_cells"),
           nb::arg("init_player_pos"), // arg for initial player position
           nb::arg("start_x"),
           nb::arg("start_y"),
           nb::arg("max_episode_length"),
           nb::arg("exec_mode"),
           nb::arg("num_worlds"),
           nb::arg("num_players"), // arg for number of players
           nb::arg("gpu_id") = -1)
        .def("step", &Manager::step)
        .def("reset_tensor", &Manager::resetTensor)
        .def("player_tensor", &Manager::playerTensor) // added new player tensor for data export
        .def("action_tensor", &Manager::actionTensor)
        .def("observation_tensor", &Manager::observationTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("done_tensor", &Manager::doneTensor)
        .def("ball_tensor", &Manager::ballTensor)
        .def("held_tensor", &Manager::heldTensor)
        .def("choice_tensor", &Manager::choiceTensor)
    ;
}

}
