#!/bin/env python
import sys
from typing import List
from data import World, Move, Person, World, Map, Shade, Tombstone, Point
from game import Game, PlayerInterface
from random import shuffle
from model import *
import torch

class Player(PlayerInterface):
    memory: PPOMemory
    train_mode: bool

    @staticmethod
    def log(*args):
        print(*args, file=sys.stderr)

    def init(self, world: World) -> None:
        Player.log("Som boh blesku a sex appealu. -idk asi Zeus")
        pass

    def backup_mem(self) -> None:
        torch.save(self.memory, "memory.pt")
    
    def load_memory(self) -> None:
        memory: PPOMemory = torch.load(r"C:\Users\Sebik\Documents\pornboj\ksp-proboj-2026-jar-foyer\players\python\memory.pt")

    def get_turn(self, world: World) -> List[Move]:
        if self.train_mode: return self.get_turn_train(world)
        
        moves = []
        for id, ant in world.alive_shades.items():
            neighbours = ant.position.get_neighbouring()
            shuffle(neighbours)
            for ngb in neighbours:
                if world.map.can_move_to(ngb):
                    moves.append(Move(id, ngb))
                    break
        return moves
    
    def get_turn_train(self, world: World) -> List[Move]:
        pass

if __name__ == "__main__":
    game = Game(Player())
    game.run()
