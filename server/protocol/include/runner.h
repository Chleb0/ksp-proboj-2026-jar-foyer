#pragma once

#include "runner_utils.h"
#include "data.h"
#include <expected>
#include <map>

class Runner{
public:
    Runner(){};

    std::expected<void, runner::Status> init_player(const std::string&, const PlayerInit&);

    std::expected<Moves, runner::Status> get_player_turn(const std::string& player_name, const PlayerTurn&);

    void kill_player(const std::string&);

    void to_observer(const ObserverTurn&);

    void init_observer(const ObserverInit&);

    /// @brief Sends scores after game ends
    /// @param scores Map of player names and their scores
    void send_scores(const std::map<std::string, double>&);

    void end_game();
private:
    /// @brief Request and read response from player
    /// @param player_name
    /// @param status
    /// @return data_stream containing player's response
    std::stringstream read_player(const std::string&, runner::Status&);
};
