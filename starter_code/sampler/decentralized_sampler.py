import numpy as np

from starter_code.sampler.sampler import Sampler
from starter_code.interfaces.interfaces import UtilityArgs
from starter_code.organism.utilities import get_second_highest_bid


class DecentralizedSampler(Sampler):
    def begin_episode(self, env):
        if not self.deterministic and self.organism.args.ado:
            self.organism.agent_dropout()
        state = super(DecentralizedSampler, self).begin_episode(env)
        return state

    def finish_episode(self, state, episode_data, env):
        episode_data = Sampler.finish_episode(self, state, episode_data, env)
        episode_data = self.assign_utilities(episode_data)
        return episode_data

    def assign_utilities(self, society_episode_data):
        for t in range(len(society_episode_data)):
            winner = society_episode_data[t].winner
            s_winner = society_episode_data[t].s_winner
            bids = society_episode_data[t].bids
            s_bids = society_episode_data[t].s_bids
            reward = society_episode_data[t].reward  # not t+1!
            start_time = society_episode_data[t].start_time
            end_time = society_episode_data[t].end_time

            w_subsociety_bids = society_episode_data[t].w_s_bids

            if t < len(society_episode_data)-1:
                next_winner = society_episode_data[t+1].winner
                next_s_winner = society_episode_data[t+1].s_winner
                next_bids = society_episode_data[t+1].bids
                next_s_bids = society_episode_data[t+1].s_bids
                next_winner_bid = next_bids[next_winner]
                next_s_winner_bid = next_s_bids[next_s_winner]

                next_w_subsociety_bids = society_episode_data[t+1].w_s_bids
                next_second_highest_bid = get_second_highest_bid(next_w_subsociety_bids, next_winner)
                next_second_highest_s_bid = get_second_highest_bid(next_s_bids, next_s_winner)
            else:
                next_winner_bid = 0
                next_s_winner_bid = 0
                next_second_highest_bid = 0
                next_second_highest_s_bid = 0

            utilities = self.organism.compute_utilities(
                UtilityArgs(
                            bids=w_subsociety_bids,
                            winner=winner,
                            next_winner_bid=next_winner_bid,
                            next_second_highest_bid=next_second_highest_bid,
                            reward=reward,
                            start_time=start_time,
                            end_time=end_time))

            for v in bids.items():
                if v[0] not in utilities:
                    utilities[v[0]] = 0

            s_utilities = self.organism.compute_utilities(
                UtilityArgs(
                    bids=s_bids,
                    winner=s_winner,
                    next_winner_bid=next_s_winner_bid,
                    next_second_highest_bid=next_second_highest_s_bid,
                    reward=reward,
                    start_time=start_time,
                    end_time=end_time))

            society_episode_data[t].set_payoffs(utilities, s_utilities)
        return society_episode_data

    def get_bids_for_episode(self, society_episode_data):
        a_ids = society_episode_data[0].bids.keys()
        s_ids = society_episode_data[0].s_bids.keys()
        episode_bids = {a_id: [] for a_id in a_ids}
        s_episode_bids = {s_id: [] for s_id in s_ids}
        for step_info in society_episode_data:
            for a_id in a_ids:
                episode_bids[a_id].append(step_info.bids[a_id])
            for s_id in s_ids:
                s_episode_bids[s_id].append(step_info.s_bids[s_id])
        return episode_bids, s_episode_bids