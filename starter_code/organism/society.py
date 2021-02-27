from collections import OrderedDict
from operator import itemgetter
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from starter_code.infrastructure.utils import from_np
from starter_code.interfaces.interfaces import DecentralizedOutput
from starter_code.rl_update.replay_buffer import StoredTransition
from starter_code.organism.agent import ActorCriticRLAgent
from starter_code.organism.domain_specific import preprocess_state_before_store
from starter_code.organism.organism import Organism
from starter_code.organism.utilities import vickrey_utilities, credit_conserving_vickrey_utilities, bucket_brigade_utilities, environment_reward_utilities


def agent_update(agent, rl_alg):
    agent.update(rl_alg)

class Society(nn.Module, Organism):
    def __init__(self, agents, unique_agents, subsocieties, unique_subsocieties, device, args):
        super(Society, self).__init__()
        self.agents = nn.ModuleList(agents)
        self.unique_agents = unique_agents  # just a set of pointers
        self.subsocieties = nn.ModuleList(subsocieties)
        self.unique_subsocieties = unique_subsocieties  # just a set of pointers
        self.agents_by_id = {agent.id: agent for agent in self.agents}  # this is the registry
        self.subsocieties_by_id = {subsociety.id: subsociety for subsociety in self.subsocieties}  # this is the registry
        self.players = args.players
        self.subsociety_n_agents = args.num_primitives*args.redundancy

        self.device = device
        self.args = args
        self.bootstrap = True
        self.discrete = True
        self.ado = args.ado

        self.transformations, self.subsocieties_transformations = self.assign_transformations()
        self.transformation_type = self.get_transformation_type(self.transformations)

        self.set_trainable(True)

    def assign_transformations(self):
        transformations = OrderedDict()
        for a_id in self.agents_by_id:
            transformations[a_id] = self.agents_by_id[a_id].transformation
        for agent in self.agents:
            agent.transformation.set_transformation_registry(transformations)
        subsocieties_transformations = OrderedDict()
        for ss_id in self.subsocieties_by_id:
            subsocieties_transformations[ss_id] = self.subsocieties_by_id[ss_id].transformation
        for subsociety in self.subsocieties:
            subsociety.transformation.set_transformation_registry(subsocieties_transformations)
        return transformations, subsocieties_transformations

    def get_transformation_type(self, transformations):
        for i, transformation in enumerate(transformations.values()):
            if i > 0:
                assert transformation.__class__.__name__ == transformation_type
            else:
                transformation_type = transformation.__class__.__name__
        return transformation_type

    def set_trainable(self, trainable):
        self.trainable = trainable

    def can_be_updated(self):
        return self.trainable

    def agent_dropout(self):
        total_agents = len(self.agents)
        num_inactive_agents = np.random.randint(low=0, high=total_agents-1)
        inactive_agent_ids = np.random.choice(a=range(total_agents), size=num_inactive_agents, replace=False)
        self._set_inactive_agents(inactive_agent_ids)

    def get_state_dict(self):
        agents_state_dict = [a.get_state_dict() for a in self.agents]
        return agents_state_dict

    def get_s_state_dict(self):
        s_state_dict = [s.get_state_dict() for s in self.subsocieties]
        return s_state_dict

    def load_state_dict(self, society_state_dict, society_subs_state_dict):
        for agent, agent_state_dict in zip(self.agents, society_state_dict):
            agent.load_state_dict(agent_state_dict)
        for s, s_state_dict in zip(self.subsocieties, society_subs_state_dict):
            s.load_state_dict(s_state_dict)

    def _set_inactive_agents(self, agent_ids):
        for agent in self.agents:
            agent.active = True

        for agent_id in agent_ids:
            assert agent_id == self.agents[agent_id].id
            self.agents[agent_id].active = False

    def get_active_agents(self):
        active_agents = []
        for agent in self.agents:
            if agent.active:
                active_agents.append(agent)
        return active_agents

    def _run_auction(self, state, deterministic):
        if self.args.clone:
            with torch.no_grad():
                dists = OrderedDict((a.id, a.policy.get_action_dist(state)) for a in self.unique_agents)
            bids = OrderedDict()
            for i in range(self.args.redundancy):
                for a in self.unique_agents:
                    bid = dists[a.id].sample().item()
                    bids[a.id + i*self.args.num_primitives] = bid

            with torch.no_grad():
                dists = OrderedDict((s.id, s.policy.get_action_dist(state)) for s in self.unique_subsocieties)
            subsocieties_bids = OrderedDict()
            for i in range(self.args.redundancy):
                for s in self.unique_subsocieties:
                    bid = dists[s.id].sample().item()
                    subsocieties_bids[s.id + i*self.players] = bid
        else:
            bids = OrderedDict([(a.id, a(state, deterministic=deterministic).item()) for a in self.get_active_agents()])
            subsocieties_bids = OrderedDict([(s.id, s(state, deterministic=deterministic).item())
                                             for s in self.subsocieties])
        return bids, subsocieties_bids

    def compute_utilities(self, utility_args):
        utility_function = dict(
            v=vickrey_utilities,
            bb=bucket_brigade_utilities,
            ccv=credit_conserving_vickrey_utilities,
            env=environment_reward_utilities)[self.args.auctiontype]
        utilities = utility_function(utility_args, self.args)
        return utilities

    def _choose_winner(self, bids, subsocieties_bids):
        if self.players > 1:
            winner_subsociety = max(subsocieties_bids.items(), key=itemgetter(1))[0]
            winner_player = self.subsocieties_by_id[winner_subsociety].transformation
        else:
            winner_subsociety = 0
            winner_player = 0
        winner = max(bids.items()[winner_player*self.subsociety_n_agents:(winner_player+1)*self.subsociety_n_agents],
                     key=itemgetter(1))[0]  # bidding only within the winning subsociety
        return winner, winner_subsociety

    def _select_action(self, winner, winner_subsociety):
        action = self.agents_by_id[winner].transformation
        if self.players == 1:
            return [action, -1]
        player = self.subsocieties_by_id[winner_subsociety].transformation
        return [action, player]

    def _get_learnable_active_agents(self):
        learnable_active_agents = [a for a in self.unique_agents if a.learnable and len(a.replay_buffer) > 0]
        learnable_subsocieties = [s for s in self.unique_subsocieties if s.learnable and len(s.replay_buffer) > 0]
        return learnable_active_agents+learnable_subsocieties

    def step_optimizer_schedulers(self, pfunc):
        for agent in self.agents:
            agent.step_optimizer_schedulers(pfunc)
        for s in self.subsocieties:
            s.step_optimizer_schedulers(pfunc)

    # def flail(self, env):
    #     bids = OrderedDict([(a.id, a.flail()) for a in self.agents])
    #     return self._produce_output(bids)

    def forward(self, state, deterministic):
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = from_np(state, 'cpu')
            bids, subsocieties_bids = self._run_auction(state, deterministic=deterministic)
        winner, winner_subsociety = self._choose_winner(bids, subsocieties_bids)
        action, player = self._select_action(winner, winner_subsociety)
        return DecentralizedOutput(action=action, player=player, winner=winner, s_winner=winner_subsociety, bids=bids, s_bids=subsocieties_bids)

    def update(self, rl_alg):
        learnable_active_agents = self._get_learnable_active_agents()

        if self.args.parallel_update:
            processes = []
            for agent in learnable_active_agents:
                p = mp.Process(target=agent_update, args=(agent, rl_alg))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        else:
            for agent in learnable_active_agents:
                agent.update(rl_alg)

    def store_path(self, path):
        processed_path = {a_id: [] for a_id in path[0].bids}  # path[0] is hacky
        s_processed_path = {s_id: [] for s_id in path[0].s_bids}
        for step in path:
            if self.args.memoryless:
                mask = 0
            else:
                mask = step.mask
            for a_id in step.bids:
                step = preprocess_state_before_store(step)
                processed_path[a_id].append(
                    StoredTransition(
                        state=step.state,
                        action=np.array([step.bids[a_id]]),
                        next_state=step.next_state,
                        mask=mask,
                        reward=step.payoffs[a_id],
                        start_time=step.start_time,
                        end_time=step.end_time,
                        current_transformation_id=step.current_transformation_id,
                        next_transformation_id=step.next_transformation_id,
                        )
                    )
            for s_id in step.s_bids:
                step = preprocess_state_before_store(step)
                processed_path[s_id].append(
                    StoredTransition(
                        state=step.state,
                        action=np.array([step.s_bids[s_id]]),
                        next_state=step.next_state,
                        mask=mask,
                        reward=step.payoffs[s_id],
                        start_time=step.start_time,
                        end_time=step.end_time,
                        current_transformation_id=step.current_transformation_id,
                        next_transformation_id=step.next_transformation_id,
                    )
                )
        for a_id in processed_path:
            self.agents_by_id[a_id].replay_buffer.add_path(processed_path[a_id])
        for s_id in s_processed_path:
            self.subsocieties_by_id[s_id].replay_buffer.add_path(s_processed_path[s_id])

    def clear_buffer(self):
        for a_id, agent in enumerate(self.agents):
            agent.clear_buffer()
        for s_id, subsociety in enumerate(self.subsocieties):
            subsociety.clear_buffer()

    def visualize_parameters(self, pfunc):
        for agent in self.agents:
            pfunc('Primitive: {}'.format(agent.id))
            agent.visualize_parameters(pfunc)
        for s in self.subsocieties:
            pfunc('Player: {}'.format(s.id))
            s.visualize_parameters(pfunc)


class BiddingPrimitive(ActorCriticRLAgent):
    def __init__(self, id_num, transformation, networks, replay_buffer, args):
        ActorCriticRLAgent.__init__(self, networks, replay_buffer, args)
        self.id = id_num
        self.transformation = transformation
        self.learnable = True
        self._active = True
        self.is_subpolicy = False

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, value):
        self._active = value

    def forward(self, state, deterministic):
        with torch.no_grad():
            bid, dist = self.policy.select_action(state, deterministic)
        return bid

    def flail(self):
        bid = np.random.uniform()  # note that the range is [0, 1]
        return bid

    def update(self, rl_alg):
        rl_alg.improve(self)
        if self.transformation.can_be_updated():
            self.transformation.update(rl_alg)

    def clear_buffer(self):
        self.replay_buffer.clear_buffer()
        self.transformation.clear_buffer()
