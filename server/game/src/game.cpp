#include "game.h"
#include "ant_spawn_phase.h"
#include "attack_phase.h"
#include "check_collisions.h"
#include "food_spawn_phase.h"
#include "gather_phase.h"
#include "razing_phase.h"
#include "constants.h"

using namespace std;

void Game::run() {
    init();
    do {
        game_loop();
    } while(!check_end());
    cleanup();
}

void Game::init() {
    // send data to observer
    //
    ObserverInit observer_init{all_players, world_map, alive_hills};
    runner.init_observer(observer_init);

    // send data to players
    vector<PlayerID> not_responding;
    PlayerInit player_init{world_map, alive_hills, PlayerID::NONE};
    for(const auto& [id, player] : all_players) {
        player_init.my_id = id;
        auto status = runner.init_player(player.get_name(), player_init);
        if (!status.has_value()) {
            cerr << "Player " << player.get_name() << " failed to initialize" << endl;
            not_responding.push_back(id);
        }
    }
    for(PlayerID id : not_responding) {
        kill_player(id);
    }
}

void Game::game_loop() {
    cerr << "STARTING TURN: " << current_turn << endl;
    // handle player moves
    //
    PlayerTurn this_turn{alive_ants, alive_hills, alive_food, 0, current_turn};
    vector<PlayerID> not_responding;
    // send data to players
    for(PlayerID player_id : alive_players) {
        Player& player = all_players[player_id];
        this_turn.food_count = player.get_food_count();

        auto response = runner.get_player_turn(player.get_name(), this_turn);
        if (!response.has_value()) {
            cerr << "Killing player " << player.get_name() << ": Failed to read turn data." << endl;
            not_responding.push_back(player_id);
            continue;
        }
        auto moves = response.value();
        handle_player_moves(player, moves);
    }

    for(PlayerID id : not_responding) {
        kill_player(id);
    }

    //  check for collisions
    //
    for(AntID ant_id : get_colliding_ants(alive_ants)) {
        kill_ant(ant_id);
    }

    // attack phase
    //
    for(AntID killed_ant : mark_killed_ants(alive_ants)){
        kill_ant(killed_ant);
    }

    // razing phase
    //
    for(auto [razed_hill, by] : mark_razed_hills(alive_ants, alive_hills)){
        kill_hill(razed_hill, by);
    }

    // spawn ants
    //
    for(PlayerID player_id : alive_players) {
        Player& player = all_players[player_id];
        for(Point p : get_new_ant_pos(player_hills[player_id], alive_hills, alive_ants)) {
            if(player.get_food_count()) {
                insert(alive_ants, Ant(p, player_id));
                player.dec_food_count();
                player.inc_ants();
            }
        }
    }

    // gather phase
    //
    for(auto [food, owner] : get_harvested_food(alive_ants, alive_food)){
        harvest_food(food, owner);
    }

    // spawn food
    //
    for(Point p : get_new_food_locations((MAX_FOOD_COUNT - alive_food.size())/2, world_map)) {
        alive_food.insert(Food(p));
    }

    // send data to observer
    //
    ObserverTurn observer_turn{alive_ants, alive_hills, alive_food, all_players, current_turn};
    runner.to_observer(observer_turn);

    current_turn++;
}

void Game::cleanup() {
    // save scores
    //
    map<std::string, double> scores;
    for (auto&& [_, player] : all_players) {
        scores[player.get_name()] = 100 * player.get_score();
    }
    runner.send_scores(scores);

    // clean up player manager
    //
    runner.end_game();
}

vector<PlayerID> check_dead_players(const set<PlayerID>& alive_players, const PlayerMap& all_players, const map<PlayerID, set<HillID>>& player_hills) {
    vector<PlayerID> dead_players;
    for(PlayerID id : alive_players) {
        const Player& player = all_players.at(id);
        if((player.get_alive_ants() == 0 && (player.get_food_count() == 0 || player_hills.at(id).size() == 0))) {
            dead_players.push_back(id);
        }
    }
    return dead_players;
}

bool Game::check_end() {
    for(PlayerID player : check_dead_players(alive_players, all_players, player_hills)) {
        cerr << "Player " << all_players[player].get_name() << " has not enough resources to continue playing" << endl;
        kill_player(player);
    }
    if(alive_players.size() == 0) {
        cerr << "Game end: No players left" << endl;
        return true;
    }
    if(alive_players.size() == 1) {
        cerr << "Game end: Last player alive" << endl;
        return true;
    }
    if(current_turn >= MAX_TURN_COUNT) {
        cerr << "Game end: Max turn count reached" << endl;
        return true;
    }
    return false;
}

void Game::kill_player(PlayerID player_id) {
    if(alive_players.find(player_id) == alive_players.end()) {
        return;
    }
    Player& player = all_players[player_id];
    runner.kill_player(player.get_name());
    cerr << "Player " << player.get_name() << " has been killed" << endl;
    alive_players.erase(player_id);
    double alive_player_count = (double) alive_players.size();
    for(PlayerID id : alive_players) {
        all_players[id].update_score(ALIVE_POINTS / alive_player_count);
    }
}

void Game::kill_ant(AntID ant_id) {
    PlayerID owner = alive_ants[ant_id].get_owner();
    all_players[owner].dec_ants();
    alive_ants.erase(ant_id);
}

void Game::kill_hill(HillID hill_id, PlayerID by) {
    Hill hill = *alive_hills.find(hill_id);
    PlayerID owner = hill.get_owner();
    player_hills[owner].erase(hill_id);
    alive_hills.erase(hill);
    all_players[owner].update_score(-POINTS_PER_HILL);
    all_players[by].update_score(RAZING_POINTS);
}

void Game::harvest_food(FoodID food_id, PlayerID player_id) {
    Food food = *alive_food.find(food_id);
    alive_food.erase(food);
    if(player_id != PlayerID::NONE) {
        all_players[player_id].inc_food_count();
    }
}
