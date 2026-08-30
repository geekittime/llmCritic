# from .alfworld.config import AlfredEnvConfig
# from .alfworld.env import AlfredTXTEnv
import importlib.util

from .bandit.config import BanditEnvConfig
from .bandit.env import BanditEnv
from .countdown.config import CountdownEnvConfig
from .countdown.env import CountdownEnv
from .sokoban.config import SokobanEnvConfig
from .sokoban.env import SokobanEnv
from .frozen_lake.config import FrozenLakeEnvConfig
from .frozen_lake.env import FrozenLakeEnv
from .metamathqa.env import MetaMathQAEnv
from .metamathqa.config import MetaMathQAEnvConfig
from .lean.config import LeanEnvConfig
from .lean.env import LeanEnv
from .sudoku.config import SudokuEnvConfig
from .sudoku.env import SudokuEnv
from .game_2048.config import Game2048EnvConfig
from .game_2048.env import Game2048Env
from .rubikscube.config import RubiksCube2x2Config
from .rubikscube.env import RubiksCube2x2Env


REGISTERED_ENVS = {
    'bandit': BanditEnv,
    'countdown': CountdownEnv,
    'sokoban': SokobanEnv,
    'frozen_lake': FrozenLakeEnv,
    # 'alfworld': AlfredTXTEnv,
    'metamathqa': MetaMathQAEnv,
    'lean': LeanEnv,
    'sudoku': SudokuEnv,
    'game_2048': Game2048Env,
    'rubikscube': RubiksCube2x2Env,
}

REGISTERED_ENV_CONFIGS = {
    'bandit': BanditEnvConfig,
    'countdown': CountdownEnvConfig,
    'sokoban': SokobanEnvConfig,
    'frozen_lake': FrozenLakeEnvConfig,
    # 'alfworld': AlfredEnvConfig,
    'metamathqa': MetaMathQAEnvConfig,
    'lean': LeanEnvConfig,
    'sudoku': SudokuEnvConfig,
    'game_2048': Game2048EnvConfig,   
    'rubikscube': RubiksCube2x2Config,
}

if importlib.util.find_spec("webshop_minimal") is not None:
    class LazyWebShopEnv:
        """Load WebShop and its Java search backend only when selected."""

        def __new__(cls, *args, **kwargs):
            from .webshop.env import WebShopEnv

            return WebShopEnv(*args, **kwargs)


    class LazyWebShopEnvConfig:
        """Defer importing webshop_minimal until a WebShop run starts."""

        def __new__(cls, *args, **kwargs):
            from .webshop.config import WebShopEnvConfig

            return WebShopEnvConfig(*args, **kwargs)


    REGISTERED_ENVS["webshop"] = LazyWebShopEnv
    REGISTERED_ENV_CONFIGS["webshop"] = LazyWebShopEnvConfig
