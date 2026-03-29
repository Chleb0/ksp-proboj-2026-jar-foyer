#pragma once

#include "types.h"
#include <map>
#include <string>

#include "glaze/glaze.hpp"

struct Player {
  public:
    Player(std::string name, PlayerID id) : name(name), id(id) {};
    Player() = default;
    std::string get_name() const { return name; }
    PlayerID get_id() const { return id; }
    double get_score() const { return score; }
    void update_score(double delta) { score += delta; }
    int get_food_count() const { return food_count; }
    void inc_food_count() { food_count++; }
    void dec_food_count() { food_count--; }
    void inc_ants() { alive_ants++; }
    void dec_ants() { alive_ants--; }
    int get_alive_ants() const { return alive_ants; }
    auto operator<=>(const Player &other) const { return id <=> other.id; }

  private:
    friend struct glz::meta<Player>;
    std::string name;
    double score = 0.0;
    int food_count = 0;
    int alive_ants = 0;
    PlayerID id;
};

template <>
struct glz::meta<Player> {
    using T = Player;
    static constexpr auto value = glz::object(
            &T::name, &T::score, &T::food_count, &T::alive_ants, &T::id);
};

typedef std::map<PlayerID, Player> PlayerMap;

/// @brief Insert player into the map and consume it
void insert(PlayerMap &, Player &&);
