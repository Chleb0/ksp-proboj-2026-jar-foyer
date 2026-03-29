#!/bin/env python
import sys
from typing import List, Set
from data import World, Move, Person, World, Map, Shade, Tombstone, Point
from collections import deque
from game import Game, PlayerInterface
from random import shuffle
from model import *
import torch

def killsAroundMe(shade: Shade, shade_positions: Dict[Point, Shade]):
    killcount = 0

    mojstrach = shade.get_fear()
    enemyfear = shade.get_enemy_fears(shade_positions)
    for shade, fear in enemyfear.items():
        if fear >= mojstrach:
            killcount += 1

    return killcount
        

def add_to_queue(visited: dict[Point, Point], q: List[Point], frm: Point, world : World):
    move_def : dict = {
                1 : Point(0,1),
                2 : Point(1,0),
                3 : Point(0,-1),
                4 : Point(-1,0)
            }

    for _, move in move_def.items():
        to = frm + move
        if to not in visited and world.map.can_move_to(to):
            visited[to] = frm
            q.append(to)

def bfs_find_person(world: World, start: Point, pipel: dict[Point, Person]) -> Point:
    if len(pipel.items()) == 0: return start

    came_from : dict[Point, Point] = {}
    came_from[start] = start

    queue : List[Point] = []
    add_to_queue(came_from, queue, start, world) 

    q_ind: int = 0
    end : Point = start
    while(len(queue) != 0):
        pnt: Point = queue[q_ind]
        if pnt in pipel:
            end = pnt
            break
        else:
            add_to_queue(came_from, queue, pnt, world)
            q_ind +=1

    if end == start:
        return start

    while(True):
        predecessor = came_from[end]
        if predecessor == start:
            return end
        else:
            end = predecessor

def getCut(board: Tensor, position: Point, vision: int) -> Tensor:

    x, y = position.x, position.y
    start_x = x
    start_y = y
    end_x = x+2*vision+1
    end_y = y+2*vision+1

    _, h, w = board.shape
    return board[:, start_x:end_x, start_y:end_y]


def getBoard(world: World, vision) -> Tensor:

        Surface = torch.tensor(boardSurface(world, vision), dtype=torch.float32)
        Enemies = torch.tensor(boardEnemies(world, vision), dtype=torch.float32)
        Friends = torch.tensor(boardFriends(world, vision), dtype=torch.float32)
        Homes = torch.tensor(boardHomes(world, vision), dtype=torch.float32)
        EnemyHomes = torch.tensor(boardEnemyHomes(world, vision), dtype=torch.float32)
        People = torch.tensor(boardPeople(world, vision), dtype=torch.float32)
        
        board = torch.stack([Surface, Enemies, Friends, Homes, EnemyHomes, People])

        return board

def boardSurface(world:World, vision:int) -> List:
    width, height = world.map.width, world.map.height
    layer = [[1 for i in range(width+vision*2)] for j in range(height+vision*2)]
    for y in range(len(layer)):
        for x in range(len(layer[y])):
            if x < vision or y < vision:
                layer[y][x] = 0
            if x >= width+vision or y >= width+vision:
                layer[y][x] = 0
    shitfpoint = Point(vision, vision)
    for voda in world.map.water_tiles:
        boardposition = voda+shitfpoint
        layer[boardposition.y][boardposition.x] = 0
    return layer

def boardEnemies(world:World, vision:int) -> List:
    width, height = world.map.width, world.map.height
    layer = [[0 for i in range(width+vision*2)] for j in range(height+vision*2)]
    shitfpoint = Point(vision, vision)
    for id, duch in world.alive_shades.items():
        if duch.owner != world.my_id:
            duchpos = duch.position
            boardposition = duchpos+shitfpoint
            layer[boardposition.y][boardposition.x] = 1
    return layer

def boardFriends(world:World, vision:int) -> List:
    width, height = world.map.width, world.map.height
    layer = [[0 for i in range(width+vision*2)] for j in range(height+vision*2)]
    shitfpoint = Point(vision, vision)
    for id, duch in world.alive_shades.items():
        if duch.owner == world.my_id:
            duchpos = duch.position
            boardposition = duchpos+shitfpoint
            layer[boardposition.y][boardposition.x] = 1
    return layer

def boardHomes(world:World, vision:int) -> List:
    width, height = world.map.width, world.map.height
    layer = [[0 for i in range(width+vision*2)] for j in range(height+vision*2)]
    shitfpoint = Point(vision, vision)
    for home in world.alive_tombstones:
        if home.owner == world.my_id:
            homepos = home.position
            boardposition = homepos+shitfpoint
            layer[boardposition.y][boardposition.x] = 1
    return layer

def boardEnemyHomes(world:World, vision:int) -> List:
    width, height = world.map.width, world.map.height
    layer = [[0 for i in range(width+vision*2)] for j in range(height+vision*2)]
    shitfpoint = Point(vision, vision)
    for home in world.alive_tombstones:
        if home.owner != world.my_id:
            homepos = home.position
            boardposition = homepos+shitfpoint
            layer[boardposition.y][boardposition.x] = 1
    return layer
            
def boardPeople(world:World, vision:int) -> List:
    width, height = world.map.width, world.map.height
    layer = [[0 for i in range(width+vision*2)] for j in range(height+vision*2)]
    shitfpoint = Point(vision, vision)
    for person in world.alive_people:
        personpos = person.position
        boardpostion = personpos+shitfpoint
        layer[boardpostion.y][boardpostion.x] = 1
    return layer


INPUT_CHANNELS = 6
OUTPUT_CHANNELS = 5
EXTRA_STAT = 1
VISION = 5
BACKUP_PATH = os.path.dirname(os.path.abspath(__file__))

class Player(PlayerInterface):
    memory: PPOMemory
    model: PPO
    train_mode: bool = True
    training_interval: int = 50

    lt_alive: int = 0
    lt_tomstones: int = 0

    shade_positions: dict[Point, Shade] = {}
    people_positions: dict[Point, Person] = {}
    our_shade = 0

    @staticmethod
    def log(*args):
        print(*args, file=sys.stderr)

    def init(self, world: World) -> None:
        #Player.log("Som boh blesku a sex appealu. -idk asi Zeus")
        
        actor_model : PPOActorCritic = PPOActorCritic(INPUT_CHANNELS, EXTRA_STAT, OUTPUT_CHANNELS)
        self.model = PPO(actor_model)

        self.memory = {
            "board": {},
            "extra": {},
            "actions": {},
            "log_probs": {},
            "rewards": {},
            "dones": {},
        }

        try:
            load_checkpoint(self.model, os.path.dirname(os.path.abspath(__file__)) + "/backup.tmp")
        except:
            pass

    def preprocess(self, world: World):
        self.people_positions = {}
        self.shade_positions = {}
        for person in world.alive_people:
            self.people_positions[person.position] = person
        
        our_shade = 0
        for _, ghost in world.alive_shades.items():
            self.shade_positions[ghost.position] = ghost
            if ghost.owner == world.my_id:
                self.our_shade+=1

    def get_turn(self, world: World) -> List[Move]:
        self.preprocess(world)
        self.preprocess(world)
        fullboard = getBoard(world, VISION)  #v+setky layers pre cel=u mapu treba orezat na vision (11x11)
        #Player.log(getCut(fullboard, Point(world.map.width-1, world.map.height-1), VISION))
        #Player.log(world.alive_tombstones)
        #Player.log("toto je pravy horny roh")
        #Player.log(self.shade_positions)
        veci = list(self.shade_positions.keys())
        vec = veci[0]
        x, y = vec.x, vec.y
        #Player.log(getCut(fullboard, Point(0, 0), VISION))
        if self.train_mode: return self.get_turn_train(world)

        #self.log(getCut(fullboard, Point(10, 10), 11))

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
        fullboard = getBoard(world, VISION) #v+setky layers pre cel=u mapu treba orezat na vision (11x11)
        #self.log(getCut(fullboard, Point(10, 10), VISION))


        moves : List[Move] = []
        shade_diff = self.shades_cnt(world) - self.lt_alive
        tom_diff = self.tom_cnt(world) - self.lt_tomstones
        fullboard = getBoard(world, VISION)

        for id, ant in world.alive_shades.items():
            if ant.owner != world.my_id: continue 

            extra: Tensor = torch.tensor(
                [self.our_shade / len(world.alive_shades)]
            )
            reward: float = self.eval_last_move(world, ant, shade_diff, tom_diff)
            if id in self.memory["board"]: self.memory["rewards"][id][-1] = reward
            else: self.setzeromem(ant.id)

            my_board = getCut(fullboard, ant.position, VISION)

            #self.log(getCut(my_board, Point(VISION, VISION), VISION))

            action, log_prob, _ = self.model.model.get_action(my_board, extra)
            volba = int(action.item())

            die = ant.will_i_die(self.shade_positions)

            # HLADIK TU DOPLN VECI KTORE RATAS
            self.memory["board"][id].append(my_board) #toto chce byt Tensor z knihovne torch, 11x11x layery ktore chceme
                                                # water/ground - 0/1
                                                # enemy/nie
                                                # friend/nie
                                                # nic/svojahrobka
                                                # nic/enem hrobka 0/1
                                                # nic/ clovek

            self.memory["extra"][id].append(extra)  #toto chcu byt tie features
            self.memory["actions"][id].append(int(action.item()))
            self.memory["log_probs"][id].append(log_prob.item())
            self.memory["rewards"][id].append(-100.0 if die else 0.0)
            self.memory["dones"][id].append(die)

            self.update_lt(world)

            #action je jeden boolean set na true vo vektore velkosti 5

            """
            Volby
            1 - hore
            2 - doprava
            3 - dole
            4 - dolava
            5 - za clovekom
            """

            move_def : dict = {
                0 : Point(0,1),
                1 : Point(1,0),
                2 : Point(0,-1),
                3 : Point(-1,0)
            }


            
            self.log("Tu som kokotko")
            if volba < 4:
                moves.append(Move(id, ant.position + move_def[volba]))
            else:
                self.log("Skusam bfs")
                moves.append(Move(id, bfs_find_person(world, ant.position, self.people_positions)))

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

    def eval_last_move(self, world: World, ghost: Shade, shade_diff: int, tom_diff: int) -> float:
        kills = 0
        dies = ghost.will_i_die(self.shade_positions)
        return kills * 10 - dies * 10 + shade_diff + tom_diff*50

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
        save_checkpoint(self.model, BACKUP_PATH)

    def check_ghost_memory(self, ghost: ShadeID, world: World):
        if ghost not in self.memory["board"]:
            self.setzeromem(ghost)
        else:
            if len(self.memory["board"][ghost]) >= self.training_interval or world.alive_shades[ghost].will_i_die(self.shade_positions):
                self.model.update(self.memory, ghost)

if __name__ == "__main__":
    game = Game(Player())
    game.run()
