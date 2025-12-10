from .aggressive import Aggressive
from .coward import Coward
from .custom import Custom
from .policy import Policy
from .random import Random
from .scanner import Scanner
from .zombie import Zombie

REGISTRY = {
    "aggressive": Aggressive,
    "coward": Coward,
    "custom": Custom,
    "random": Random,
    "scanner": Scanner,
    "zombie": Zombie,
}
