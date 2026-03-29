#!/bin/env python
import sys
from typing import List
from data import World, Move, Person, World, Map, Shade, Tombstone, Point
from collections import deque
from game import Game, PlayerInterface
from random import shuffle
from model import *
import torch

def bfs(world: World, start:Point, end:Point, directions:List) -> List:
    width, height = world.map.width, world.map.height
    layer = [[1 for i in range(width)] for j in range(height)]
    for voda in world.map.water_tiles:
        layer[voda.x][voda.y] = 0
    
    q = deque([start, []])
    visited = set(start)

    if start == end:
        return []
    
    endnotfound = False

    while endnotfound == False:
        new = q.popleft()
        neighbours = new.get_neighbouring()
        for neighbour in neighbours:
            
            if neighbour == end:
                endnotfound = True



        



def getBoard(world: World) -> Tensor:

        Surface = torch.tensor(boardSurface(world), dtype=torch.bool)
        Enemies = torch.tensor(boardEnemies(world), dtype=torch.bool)
        Friends = torch.tensor(boardFriends(world), dtype=torch.bool)
        Homes = torch.tensor(boardHomes(world), dtype=torch.bool)
        EnemyHomes = torch.tensor(boardEnemyHomes(world), dtype=torch.bool)
        People = torch.tensor(boardPeople(world), dtype=torch.bool)
        
        board = torch.stack([Surface, Enemies, Friends, Homes, EnemyHomes, People])

        return board

def boardSurface(world:World) -> List:
    width, height = world.map.width, world.map.height
    layer = [[1 for i in range(width)] for j in range(height)]
    for voda in world.map.water_tiles:
        layer[voda.x][voda.y] = 0
    return layer

def boardEnemies(world:World) -> List:
    width, height = world.map.width, world.map.height
    layer = [[0 for i in range(width)] for j in range(height)]
    for duch in world.alive_shades:
        if duch.id != world.my_id:
            layer[duch.x][duch.y] = 1

def boardFriends(world:World) -> List:
    width, height = world.map.width, world.map.height
    layer = [[0 for i in range(width)] for j in range(height)]
    for duch in world.alive_shades:
        if duch.id == world.my_id:
            layer[duch.x][duch.y] = 1

def boardHomes(world:World) -> List:
    width, height = world.map.width, world.map.height
    layer = [[0 for i in range(width)] for j in range(height)]
    for home in world.alive_tombstones:
        if home.id == world.my_id:
            layer[home.x][home.y] = 1

def boardEnemyHomes(world:World) -> List:
    width, height = world.map.width, world.map.height
    layer = [[0 for i in range(width)] for j in range(height)]
    for home in world.alive_tombstones:
        if home.id != world.my_id:
            layer[home.x][home.y] = 1
            
def boardPeople(world:World) -> List:
    width, height = world.map.width, world.map.height
    layer = [[0 for i in range(width)] for j in range(height)]
    for person in world.alive_people:
        layer[person.x][person.y] = 1


def makeMove():
    



class Player(PlayerInterface):
    memory: PPOMemory
    model: PPO
    train_mode: bool = True
    training_interval: int = 50
    interval_counter: int = 0

    lt_alive: int = 0
    lt_tomstones: int = 0

    shade_positions: Dict[Point, Shade] = {}

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
                "board": {},
                "extra": {},
                "actions": {},
                "log_probs": {},
                "rewards": {},
                "dones": {}
            }
        

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
        self.interval_counter += 1

        moves = []
        for id, ant in world.alive_shades.items():
            
            board : Tensor
            fullboard = getBoard(game.world) #v+setky layers pre cel=u mapu treba orezat na vision (11x11)
            
            # HLADIK TU DOPLN VECI KTORE RATAS
            self.memory["board"][id].append(board) #toto chce byt Tensor z knihovne torch, 11x11x layery ktore chceme
                                                # water/ground - 0/1
                                                # enemy/nie
                                                # friend/nie
                                                # nic/svojahrobka
                                                # nic/enem hrobka 0/1
                                                # nic/ clovek

            self.memory["extra"][id].append(extra)  #toto chcu byt tie features
            self.memory["actions"][id].append(action.item())
            self.memory["log_probs"][id].append(log_prob.item())
            self.memory["rewards"][id].append(reward)
            self.memory["dones"][id].append(done)

            self.update_lt(world)

            akcia: List

            #action je jeden boolean set na true vo vektore velkosti 5

            #nech sa rozhodne medzi
            # chod k clovekovi najblizsiemu
            # chod niekam smer

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
        
        for id, ant in world.alive_shades.items():
            c

            # HLADIK TU DOPLN VECI KTORE RATAS
            self.memory["board"][id].append(board) #toto chce byt Tensor z knihovne torch, 11x11x layery ktore chceme
            self.memory["extra"][id].append(extra)  #toto chcu byt tie features
            self.memory["actions"][id].append(action.item())
            self.memory["log_probs"][id].append(log_prob.item())
            self.memory["rewards"][id].append(reward)
            self.memory["dones"][id].append(done)

    def setzeromem(self, ghost: ShadeID):
        self.memory["board"][ghost] = []
        self.memory["extra"][ghost] = []
        self.memory["actions"][ghost] = []
        self.memory["log_probs"][ghost] = []
        self.memory["rewards"][ghost] = []
        self.memory["dones"][ghost] = []

    def train_one_ghost(self, ghost: ShadeID):
        self.model.update(self.memory, ghost)
        self.setzeromem(ghost)

    def check_ghost_memory(self, ghost: ShadeID, world: World):
        if ghost not in self.memory["board"]:
            self.setzeromem(ghost)
        else:
            if len(self.memory["board"][ghost]) >= self.training_interval or world.alive_shades[ghost].will_i_die(self.shade_positions):
                self.model.update(self.memory, ghost)
                

    def eval_ghost_turn(self, shade: Shade, world: World) -> float:
        pass
        

if __name__ == "__main__":
    game = Game(Player())
    game.run()
