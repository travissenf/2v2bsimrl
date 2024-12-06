#include "mgr.hpp"

#include <madrona/macros.hpp>
#include <madrona/py/bindings.hpp>

namespace madsimple {

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
                            nb::ndarray<float, nb::shape<-1, 6>,
                                nb::c_contig, nb::device::cpu> init_player_pos, // new input array for initial player pos
                            int64_t max_episode_length,
                            madrona::py::PyExecMode exec_mode,
                            int64_t num_worlds,
                            int64_t num_players, // given number of players (need to decide if we include all players or just playing players)
                            int64_t gpu_id) {


            
            Player *players = setupPlayerData(init_player_pos, num_players); // call our player data setup function

            new (self) Manager(Manager::Config {
                .maxEpisodeLength = (uint32_t)max_episode_length,
                .execMode = exec_mode,
                .numWorlds = (uint32_t)num_worlds,
                .numPlayers = (uint32_t)num_players, // new, passing in num_players to config
                .gpuID = (int)gpu_id,
            }, CourtState { // new, passing in our court state to the manager
                .players = players,
                .numPlayers = (int32_t)num_players
            });

            delete[] players;
        }, // nb::arg("walls"),
        //    nb::arg("rewards"),
        //    nb::arg("end_cells"),
           nb::arg("init_player_pos"), // arg for initial player position
        //    nb::arg("start_x"),
        //    nb::arg("start_y"),
           nb::arg("max_episode_length"),
           nb::arg("exec_mode"),
           nb::arg("num_worlds"),
           nb::arg("num_players"), // arg for number of players
           nb::arg("gpu_id") = -1)
        .def("step", &Manager::step)
        .def("player_tensor", &Manager::playerTensor) // added new player tensor for data export
        .def("action_tensor", &Manager::actionTensor)
        .def("ball_tensor", &Manager::ballTensor)
        .def("held_tensor", &Manager::heldTensor)
        .def("passing_data_tensor", &Manager::passingDataTensor)
        .def("scorecard_tensor", &Manager::gameStateTensor)
        .def("choice_tensor", &Manager::choiceTensor)
    ;
}

}
