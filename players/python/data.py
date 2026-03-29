from __future__ import annotations
import math

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Set, List
from constants import SHADE_BATTLE_RADIUS2

ShadeID = int
PlayerID = int
TombstoneID = int
PersonID = int

# Porovnavanie funguje ako porovnavanie tuplov (x, y)
@dataclass(frozen=True, order=True)
class Point:
    x: int
    y: int

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        return Point(self.x - other.x, self.y - other.y)

    def dist2(self, other: Point) -> int:
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2

    def manhattan_dist(self, other: Point) -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def get_neighbouring(self) -> List[Point]:
        return [
            Point(self.x + 1, self.y),
            Point(self.x, self.y + 1),
            Point(self.x - 1, self.y),
            Point(self.x, self.y - 1)
        ]

    def get_visible(self) -> List[Point]:
        """
        Vrati policka, na ktore by dusa z tejto pozicie dovidela.
        """
        view = []
        max_abs_delta = math.ceil(math.sqrt(SHADE_BATTLE_RADIUS2))
        for i in range(-max_abs_delta, max_abs_delta + 1):
            for j in range(-max_abs_delta, max_abs_delta + 1):
                p = Point(self.x + i, self.y + j)
                if self.dist2(p) <= SHADE_BATTLE_RADIUS2:
                    view.append(p)
        return view


@dataclass(frozen=True)
class Move:
    id: ShadeID
    target: Point


@dataclass
class Map:
    width: int
    height: int
    water_tiles: FrozenSet[Point] = field(default_factory=frozenset)
    tombstone_positions: FrozenSet[Point] = field(default_factory=frozenset)

    def is_inside(self, p: Point) -> bool:
        return 0 <= p.x < self.width and 0 <= p.y < self.height

    def can_move_to(self, p: Point) -> bool:
        return self.is_inside(p) and p not in self.water_tiles


@dataclass(frozen=True)
class Shade:
    position: Point
    owner: PlayerID
    id: ShadeID

    def get_fear(self, shade_positions: Dict[Point, Shade]) -> int:
        """
        Vrati pocet dusi z nepriatelskych timov vo vyhlade tejto duse.
        """
        fear = 0
        for pos in self.position.get_visible():
            if pos in shade_positions and shade_positions[pos].owner != self.owner:
                fear += 1
        return fear

    def get_enemy_fears(self, shade_positions: Dict[Point, Shade]) -> Dict[Shade, int]:
        """
        Vrati slovnik so strachmi dusi z nepriatelskych timov v dohlade tejto duse.
        """
        fears = {}
        for pos in self.position.get_visible():
            if pos in shade_positions and shade_positions[pos].owner != self.owner:
                fears[shade_positions[pos]] = shade_positions[pos].get_fear(shade_positions)
        return fears

    def will_i_die(self, shade_positions: Dict[Point, Shade]) -> bool:
        """
        True, ak sa dusa nedozije dalsieho kola, inac False.
        """
        enemy_fears = self.get_enemy_fears(shade_positions)
        mn_enemy_fear = min(enemy_fears.values()) if enemy_fears else math.inf
        return mn_enemy_fear <= self.get_fear(shade_positions)


@dataclass(frozen=True)
class Tombstone:
    position: Point
    owner: PlayerID
    id: TombstoneID


# Porovnavanie funguje podla porovnavania pozicii
@dataclass(frozen=True, order=True)
class Person:
    position: Point
    id: PersonID


@dataclass
class World:
    my_id: PlayerID
    map: Map
    converted_count: int = 0
    turn: int = 0
    alive_shades: Dict[ShadeID, Shade] = field(default_factory=dict)
    alive_tombstones: Set[Tombstone] = field(default_factory=set)
    alive_people: Set[Person] = field(default_factory=set)
