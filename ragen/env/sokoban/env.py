import gym
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
import numpy as np
from .utils import (
    generate_room,
    collect_entity_coordinates,
    format_coordinate_render,
)
# from gym_sokoban.envs.sokoban_env.utils import generate_room
from ragen.env.base import BaseDiscreteActionEnv
from ragen.env.sokoban.config import SokobanEnvConfig
from ragen.utils import all_seed

class SokobanEnv(BaseDiscreteActionEnv, GymSokobanEnv):
    def __init__(self, config=None, **kwargs):
        self.config = config or SokobanEnvConfig()
        self.GRID_LOOKUP = self.config.grid_lookup
        self.ACTION_LOOKUP = self.config.action_lookup
        self.search_depth = self.config.search_depth
        # Gymnasium supports ``start`` while Gym 0.21 does not.  Keep the
        # public action IDs 1..4 where possible and fall back cleanly in the
        # older ragen environment used by some workers.
        try:
            action_space = gym.spaces.discrete.Discrete(4, start=1)
        except TypeError:
            action_space = gym.spaces.discrete.Discrete(4)
        self.render_mode = self.config.render_mode
        self.observation_format = self.config.observation_format

        BaseDiscreteActionEnv.__init__(self)
        GymSokobanEnv.__init__(
            self,
            dim_room=self.config.dim_room, 
            max_steps=self.config.max_steps,
            num_boxes=self.config.num_boxes,
            **kwargs
        )
        # Gym-Sokoban overwrites ``action_space`` with its native 0-based
        # space during construction.  Restore the RAGEN action IDs after the
        # superclass has initialized the room; otherwise metadata and the
        # actual 1..4 action lookup disagree on Gymnasium versions.
        self.ACTION_SPACE = action_space
        self.action_space = action_space

    def reset(self, seed=None, mode=None):
        try:
            with all_seed(seed):
                self.room_fixed, self.room_state, self.box_mapping, action_sequence = generate_room(
                    dim=self.dim_room,
                    num_steps=self.num_gen_steps,
                    num_boxes=self.num_boxes,
                    search_depth=self.search_depth
                )
            self.num_env_steps, self.reward_last, self.boxes_on_target = 0, 0, 0
            self.player_position = np.argwhere(self.room_state == 5)[0]
            return self.render()
        except (RuntimeError, RuntimeWarning) as e:
            next_seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
            return self.reset(next_seed)
        
    def step(self, action: int):
        # Gym-Sokoban may update ``player_position`` in place.  Copy the
        # coordinate before stepping so action-effectiveness reflects the
        # actual transition rather than two references to the same array.
        previous_pos = np.array(self.player_position, copy=True)
        result = GymSokobanEnv.step(self, action)
        if len(result) == 5:
            _, reward, terminated, truncated, raw_info = result
            done = bool(terminated or truncated)
        else:
            _, reward, done, raw_info = result
            terminated = bool(done)
            truncated = False
        next_obs = self.render()
        action_effective = not np.array_equal(previous_pos, self.player_position)
        # Preserve the environment reward/termination fields for metrics and
        # make the success contract explicit for EnvStateManager.
        info = dict(raw_info or {})
        info.update({
            "action_is_effective": bool(action_effective),
            "action_is_valid": True,
            "success": bool(self.boxes_on_target == self.num_boxes),
            "raw_reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })
        return next_obs, reward, done, info

    def render(self, mode=None):
        if mode in {'grid', 'coord', 'grid_coord'}:
            return self._render_text(mode)

        render_mode = mode if mode is not None else self.render_mode
        if render_mode == 'text':
            return self._render_text(self.observation_format)
        if render_mode == 'rgb_array':
            return self.get_image(mode='rgb_array', scale=1)
        raise ValueError(f"Invalid mode: {render_mode}")

    def _render_text(self, observation_format: str) -> str:
        if observation_format == 'grid':
            room = np.where((self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state)
            return '\n'.join(''.join(self.GRID_LOOKUP.get(cell, "?") for cell in row) for row in room.tolist())
        if observation_format == 'coord':
            entity_coords = collect_entity_coordinates(self.room_state, self.room_fixed)
            return format_coordinate_render(entity_coords, self.dim_room)
        if observation_format == 'grid_coord':
            entity_coords = collect_entity_coordinates(self.room_state, self.room_fixed)
            return "Coordinates: \n" + format_coordinate_render(entity_coords, self.dim_room) + "\n" + "Grid Map: \n" + self._render_text('grid')
        raise ValueError(f"Invalid observation_format: {observation_format}")
    
    def get_all_actions(self):
        return list([k for k in self.ACTION_LOOKUP.keys()])
    
    def close(self):
        self.render_cache = None
        super(SokobanEnv, self).close()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    config = SokobanEnvConfig(dim_room=(6, 6), num_boxes=1, max_steps=100, search_depth=10)
    env = SokobanEnv(config)
    for i in range(10):
        print(env.reset(seed=1010 + i))
        print()
    while True:
        keyboard = input("Enter action: ")
        if keyboard == 'q':
            break
        action = int(keyboard)
        assert action in env.ACTION_LOOKUP, f"Invalid action: {action}"
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
    np_img = env.get_image('rgb_array')
    # save the image
    plt.imsave('sokoban1.png', np_img)
