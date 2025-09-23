from enum import Enum

class Role(str, Enum): TOP="top"; JUNGLE="jungle"; MIDDLE="middle"; BOTTOM="bottom"; SUPPORT="support"
ROLES = [r.value for r in Role]

class Strategy(str, Enum): 
    MAXIMIZE="Maximize"
    MINIMAX="Minimax"

STRATEGIES = [s.value for s in Strategy]