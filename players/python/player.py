#!/bin/env python
import sys
from typing import List
from data import World, Move, Person, World, Map, Shade, Tombstone, Point
from game import Game, PlayerInterface
from random import shuffle


class Player(PlayerInterface):
    @staticmethod
    def log(*args):
        print(*args, file=sys.stderr)

    def init(self, world: World) -> None:
        Player.log("Som boh blesku a sex appealu. -idk asi Zeus")
        pass

    def get_turn(self, world: World) -> List[Move]:
        moves = []
        for id, ant in world.alive_shades.items():
            neighbours = ant.position.get_neighbouring()
            shuffle(neighbours)
            for ngb in neighbours:
                if world.map.can_move_to(ngb):
                    moves.append(Move(id, ngb))
                    break
        return moves

if __name__ == "__main__":
    game = Game(Player())
    game.run()
