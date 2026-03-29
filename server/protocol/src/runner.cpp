#include "runner.h"
#include "data.h"
#include "runner_utils.h"
#include "format.h"
#include <expected>

using namespace std;

stringstream Runner::read_player(const string& player, runner::Status& status){
    runner::sendCommand("READ PLAYER", {player}, "");
    return runner::read(status);
}

expected<void, runner::Status> Runner::init_player(const std::string& player_name, const PlayerInit &state) {
    auto message = format::init_player(state);
    runner::sendCommand("TO PLAYER", {player_name}, message);
    auto status = runner::read_status();
    read_player(player_name, status);

    if (status != runner::Status::Ok) {
        return unexpected(status);
    }
    return {};
}

void Runner::init_observer(const ObserverInit& state) {
    runner::sendCommand("TO OBSERVER", {}, format::init_observer(state));
    auto status = runner::read_status();

    if (status != runner::Status::Ok) {
        std::cerr << "Failed to init observer" << endl;
    }
}

std::expected<Moves, runner::Status> Runner::get_player_turn(const std::string& player_name, const PlayerTurn& state) {
    auto message = format::to_player(state);
    runner::sendCommand("TO PLAYER", {player_name}, message);
    auto status = runner::read_status();

    auto response = read_player(player_name, status);
    if (status != runner::Status::Ok) {
        return std::unexpected(status);
    }

    auto moves = Moves::from_sstream(response);
    if (!moves.has_value()) {
        std::cerr << player_name << ": Failed to parse turn" << endl;
        return std::unexpected(runner::Status::Error);
    }
    return *moves;
}

void Runner::to_observer(const ObserverTurn& state) {
    runner::sendCommand("TO OBSERVER", {}, format::to_observer(state));
    auto status = runner::read_status();

    if (status != runner::Status::Ok) {
        std::cerr << "Failed to init observer" << endl;
    }
}

void Runner::send_scores(const map<string, double>& scores){
    cerr << "Sending scores" << endl;
    std::stringstream payload;

    for(auto it = scores.begin();it != scores.end();it++){
        if(it != scores.begin())
            payload << "\n";
        payload << it->first<< " " << it->second;
    }

    runner::sendCommand("SCORES", payload.str());
    auto status = runner::read_status();

    if(status != runner::Status::Ok){
        cerr << "Could not send scores to runner" << endl;
    }
}

void Runner::end_game() {
    runner::sendCommand("END");
    auto status = runner::read_status();

    if(status != runner::Status::Ok){
        cerr << "Could not end game" << endl;
    }
}

void Runner::kill_player(const std::string& player_name) {
    runner::sendCommand("KILL PLAYER", {player_name}, "");
    auto status = runner::read_status();

    if(status != runner::Status::Ok){
        cerr << "Could not kill player " << player_name << endl;
    }
}
