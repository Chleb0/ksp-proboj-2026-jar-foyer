#include "game.h"
#include "config.h"
#include "rng.h"
#include "runner_utils.h"
#include "constants.h"

#include <algorithm>
#include <glaze/beve/header.hpp>
#include <random>
#include <sstream>
#include <sys/stat.h>

int MAX_FOOD_COUNT;
double POINTS_PER_HILL;
double RAZING_POINTS;

using namespace std;

void set_score_constants(int spawned_hills_count, int player_count) {
    double ratio = (double) spawned_hills_count / player_count;

    POINTS_PER_HILL = 1.0 / ratio;
    RAZING_POINTS = 2.0 / ratio;
}

void set_max_food_count(int player_count) {
    std::uniform_int_distribution<> dist(player_count, 3 * player_count);
    MAX_FOOD_COUNT = dist(rng);
    cerr << "Max food count is: " << MAX_FOOD_COUNT << endl;
}

optional<Game> Game::load_config() {
    runner::Status status;
    auto config = runner::read(status);

    vector<PlayerConfig> players;
    string line, player_name;
    getline(config, line);

    stringstream player_names_line(std::move(line));
    while (player_names_line >> player_name) {
        players.push_back(PlayerConfig(std::move(player_name)));
    }

    getline(config, line);
    auto map_result = Map::from_ppm(line);
    if(!map_result.has_value()) {
        return nullopt;
    }
    Map world_map = std::move(map_result.value());

    Game game(
        std::move(world_map)
    );

    int player_count = players.size();
    int number_of_hills = game.world_map.get_hill_positions().size();

    if (number_of_hills < player_count) {
        cerr << "Not enough hills for players!" << endl;
        return nullopt;
    }

    int hills_per_player = number_of_hills / player_count;
    int used_hills = hills_per_player * player_count;

    vector<Point> selected_hills;
    sample(game.world_map.get_hill_positions().begin(),
        game.world_map.get_hill_positions().end(),
        back_inserter(selected_hills), used_hills, rng);
    shuffle(selected_hills.begin(), selected_hills.end(), rng);

    set_max_food_count(player_count);
    set_score_constants(used_hills, player_count);

    int current_hill = 0;
    // setup alive players
    for(const PlayerConfig& conf : players) {
        Player player = Player(conf.name, conf.id);
        for(int i = 0; i < hills_per_player; i++,current_hill++) {
            Point pos = selected_hills[current_hill];
            // assign new hill and ant to player
            Hill hill(pos, conf.id);
            game.player_hills[conf.id].insert(hill.get_id());
            game.alive_hills.insert(std::move(hill));
            insert(game.alive_ants, Ant(pos, conf.id));
            player.inc_ants();
            player.update_score(POINTS_PER_HILL);
        }
        game.alive_players.insert(player.get_id());
        insert(game.all_players, std::move(player));
    }

    return game;
}
