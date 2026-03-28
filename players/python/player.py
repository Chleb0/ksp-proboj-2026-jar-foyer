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
    train_mode: bool = True

    lt_alive: int = 0
    lt_tomstones: int = 0

    @staticmethod
    def log(*args):
        print(*args, file=sys.stderr)

    def init(self, world: World) -> None:
        Player.log("Som boh blesku a sex appealu. -idk asi Zeus")
        pass

    def backup_mem(self) -> None:
        torch.save(self.memory, "memory.pt")
    
    def load_memory(self) -> None:
        try:
            memory: PPOMemory = torch.load(r"C:\Users\Sebik\Documents\pornboj\ksp-proboj-2026-jar-foyer\players\python\memory.pt")
            Player.log("Dojebal som načítavanie dát kokotko")
        except:
            memory: PPOMemory = {
                "board": [],
                "extra": [],
                "actions": [],
                "log_probs": [],
                "rewards": [],
                "dones": []
            }

    def generate_layers(self, world: World, point: Point, vision: int) -> List:
        topright = (point.x-vision, point.y-vision)
        

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
        moves = []
        for id, ant in world.alive_shades.items():
            
            
            # HLADIK TU DOPLN VECI KTORE RATAS
            self.memory["board"].append(board) #toto chce byt Tensor z knihovne torch, 11x11x layery ktore chceme
            self.memory["extra"].append(extra)  #toto chcu byt tie features
            self.memory["actions"].append(action.item())
            self.memory["log_probs"].append(log_prob.item())
            self.memory["rewards"].append(reward)
            self.memory["dones"].append(done)

            self.update_lt(world)

        return moves

    def update_lt(self, world: World) -> None:
        self.lt_alive = self.shades_cnt(world)

        self.lt_tomstones = self.tom_cnt(world)

    def shades_cnt(self, world: World) -> int:
        out = 0
        for id, ant in world.alive_shades.items():
            if ant.owner == world.my_id:
                out+=1
        return out

    def tom_cnt(self, world: World) -> int:
        out = 0
        for tom in world.alive_tombstones:
            if tom.owner == world.my_id:
                out += 1
        return out

    def eval_last_move(self, world: World):
        shade_diff = self.shades_cnt(world) - self.lt_alive
        tom_diff = self.tom_cnt(world) - self.lt_tomstones

    
    def eval_ghost_turn(self, shade: Shade, world: World) -> float:
        pass
        

if __name__ == "__main__":
    game = Game(Player())
    game.run()
