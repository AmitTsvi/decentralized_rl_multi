
from collections import namedtuple
import gym
from gym.wrappers.time_limit import TimeLimit
import gym_minigrid
from gym_minigrid.wrappers import ImgObsWrapper
import pprint
import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('./open_spiel/build/python')
import pyspiel


from mnist.mnist_env import mnist_loss_01, MentalRotation
from starter_code.environment.envs import OneStateOneStepKActionEnv, OneHotChainK, Duality
from starter_code.environment.computation_envs import ComputationEnv, VisualComputationEnv


class EnvInfo():
    def __init__(self, env_name, env_type, reward_shift=0, reward_scale=1, **kwargs):
        self.env_name = env_name
        self.env_type = env_type

        # set default
        self.reward_shift = reward_shift
        self.reward_scale = reward_scale

        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __str__(self):
        return str(self.__dict__)

    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        return str(self.__dict__)


def build_env_infos(d):
    count = 0
    tranposed_dict = dict()
    for key, subdict in d.items():
        for subkey in subdict:
            tranposed_dict[subkey] = EnvInfo(env_name=subkey, env_type=key, **subdict[subkey])
            count += 1
    assert len(tranposed_dict.keys()) == count
    return tranposed_dict


def simplify_name(names):
    return '_'.join(''.join(x for  x in name if not x.islower()) for name in names)


class RewardNormalize(gym.RewardWrapper):
    def __init__(self, env, scale=1, shift=0):
        super().__init__(env)
        self.scale = scale
        self.shift = shift

    def reward(self, r):
        return (r - self.shift) * self.scale


class GymRewardNormalize(RewardNormalize):
    def __init__(self, env, scale=1, shift=0):
        if isinstance(env, TimeLimit):
            self._max_episode_steps = env._max_episode_steps
            # unwrap the TimeLimit
            RewardNormalize.__init__(self, env.env, scale, shift)
        else:
            assert False


class MiniGridRewardNormalize(RewardNormalize):
    def __init__(self, env, scale=1, shift=0):
        RewardNormalize.__init__(self, env, scale, shift)
        self.max_steps = env.env.max_steps


class BoxPushEnvWrapper:
    def __init__(self, scale=1, shift=0):
        self.scale = scale
        self.shift = shift
        self.game = pyspiel.load_game("coop_box_pushing_serial",
                                      {"fully_observable":pyspiel.GameParameter(True),
                                       "horizon":pyspiel.GameParameter(100)})
        self.state = self.game.new_initial_state()
        self._max_episode_steps = self.game.max_game_length()
        self.env_seed = 0

    def step(self, id_num, player=-1):
        if player == 0:
            self.state.apply_actions([id_num, 3])
        else:
            self.state.apply_actions([3, id_num])
        # For a deterministic game
        self.state.apply_action(0)  # Success for player1 action
        self.state.apply_action(0)  # Success for player2 action
        self.state.apply_action(2)  # A constant order of actions applied (e.g. when both try to move to the same spot)

        reward = self.state.rewards()[0]
        done = self.state.is_terminal()
        info = {}
        return self.state, reward, done, info  # maybe need to send copy.copy(self.state)?

    def reset(self):
        return self.game.new_initial_state()

    def reward(self, r):
        return (r - self.shift) * self.scale

    def seed(self, s):
        self.env_seed = s

    def render(self, mode='human'):
        if mode == 'rgb_array':
            t = self.state.observation_tensor(0)
            t = torch.tensor(t)
            t = t.reshape(11, 8, 8).detach().numpy()
            norm_array = lambda a: (255*(a - np.min(a))/np.ptp(a)).astype('uint8')
            rgb_frame = np.zeros((8, 8, 3,), dtype='uint8')
            rgb_frame[:,:,0] = norm_array(t[1])
            rgb_frame[:,:,1] = norm_array(t[2])
            rgb_frame[:,:,2] = norm_array(np.sum(t[3:10], 0))
            return rgb_frame



class EnvRegistry():
    def __init__(self):
        self.envs_type_name = {
            'gym': {
                'Hopper-v2': dict(),
            },
            'mg': {
                'BabyAI-GreenTwoRoomTest-v0': dict(),
                'BabyAI-BlueTwoRoomTest-v0': dict(),

                'BabyAI-RedGoalTwoRoomTest-v0': dict(),
                'BabyAI-GreenGoalTwoRoomTest-v0': dict(),
                'BabyAI-BlueGoalTwoRoomTest-v0': dict(),
            },
            'vcomp': {
                'MentalRotation': dict(constructor=lambda: VisualComputationEnv(
                    dataset=MentalRotation(),
                    loss_fn=mnist_loss_01,
                    max_steps=2)),
            },
            'tab': {
                'Bandit': dict(constructor=lambda: OneStateOneStepKActionEnv(4)),
                'Chain': dict(constructor=lambda: OneHotChainK(6)),
                'Duality': dict(constructor=lambda: Duality(absorbing_reward=-1, asymmetric_coeff=1, big_reward=0.5, small_reward=0.3)), 
            },
            'open_spiel': {
                'coop_box_pushing': dict(),
            }
        }

        self.typecheck(self.envs_type_name)
        self.env_infos = build_env_infos(self.envs_type_name)

    def typecheck(self, d):
        assert type(d) == dict
        for key, value in d.items():
            assert type(value) == dict or type(value) == set

    def get_env_constructor(self, env_name):
        env_type = self.get_env_type(env_name)
        if env_type == 'mg':
            constructor = lambda: MiniGridRewardNormalize(
                ImgObsWrapper(gym.make(env_name)),
                scale=self.env_infos[env_name].reward_scale,
                shift=self.env_infos[env_name].reward_shift)
        elif env_type == 'gym':
            constructor = lambda: GymRewardNormalize(
                gym.make(env_name),
                scale=self.env_infos[env_name].reward_scale,
                shift=self.env_infos[env_name].reward_shift)
        elif env_type in ['tab', 'vcomp']:
            constructor = self.env_infos[env_name].constructor
        elif env_type == 'open_spiel':
            constructor = lambda: BoxPushEnvWrapper(
                scale=self.env_infos[env_name].reward_scale,
                shift=self.env_infos[env_name].reward_shift)
        else:
            assert False

        return constructor

    def get_env_type(self, env_name):
        return self.env_infos[env_name].env_type

    def get_reward_normalization_info(self, env_name):
        env_info = self.env_infos[env_name]
        return dict(reward_shift=env_info.reward_shift, reward_scale=env_info.reward_scale)
