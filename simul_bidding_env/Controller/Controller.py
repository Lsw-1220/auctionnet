import gin
import numpy as np
import os
from simul_bidding_env.strategy.player_agent_wrapper import PlayerAgentWrapper
from simul_bidding_env.PvGenerator.NeurIPSPvGen import NeurIPSPvGen
from simul_bidding_env.Environment.BiddingEnv import BiddingEnv
from simul_bidding_env.PvGenerator.ModelPvGen import ModelPvGenerator
from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy
from simul_bidding_env.strategy.onlinelp_bidding_strategy import OnlineLpBiddingStrategy
from simul_bidding_env.strategy.bc_bidding_strategy import BcBiddingStrategy
from simul_bidding_env.strategy.mbrl_combomicro_bidding_strategy import MbrlComboMicroBiddingStrategy
from simul_bidding_env.strategy.bcq_bidding_strategy import BcqBiddingStrategy
from simul_bidding_env.strategy.cql_bidding_strategy import CqlBiddingStrategy
from simul_bidding_env.strategy.td3_bc_bidding_strategy import TD3_BCBiddingStrategy
from simul_bidding_env.strategy.iql_bidding_strategy import IqlBiddingStrategy
from simul_bidding_env.strategy.mbrl_mopo_bidding_strategy import MbrlMopoBiddingStrategy


@gin.configurable
class Controller:
    def __init__(self, player_index: int = 0, player_agent=None, num_tick=24, num_agent_category=6, num_category=5,pv_num=500000,pv_generator_type="neuripsPvGen"):
        self.player_index = player_index
        self.num_agent_category = num_agent_category
        self.num_category = num_category
        self.num_agent = self.num_agent_category * self.num_category
        self.num_tick = num_tick
        self.pv_num = pv_num
        self.pv_generator_type=pv_generator_type
        self.agent_list = self.initialize_agents()
        self.budget_list = self.calculate_budget()
        self.category = np.arange(self.num_agent) // self.num_agent_category
        self.cpa_constraint_list = self.get_cpa_constraints()
        self.player_agent = player_agent
        self.agents = self.load_agents()
        self.pvGenerator = self.load_pv_generator()
        self.biddingEnv = self.load_bidding_env()

    def initialize_agents(self) -> list:
        """Initialize agents"""
        agents = []
        for i in range(self.num_category):
            if i % 2 == 0:
                agents.extend([
                    PidBiddingStrategy(exp_tempral_ratio=np.ones(48)),
                    IqlBiddingStrategy(),
                    TD3_BCBiddingStrategy(),
                    OnlineLpBiddingStrategy(episode=i),
                    OnlineLpBiddingStrategy(episode=i + 1),
                    CqlBiddingStrategy(),
                    BcBiddingStrategy(),
                    MbrlMopoBiddingStrategy()
                ])
            else:
                agents.extend([
                    PidBiddingStrategy(exp_tempral_ratio=np.ones(48)),
                    BcqBiddingStrategy(),
                    MbrlMopoBiddingStrategy(),
                    OnlineLpBiddingStrategy(episode=i),
                    OnlineLpBiddingStrategy(episode=i + 1),
                    TD3_BCBiddingStrategy(),
                    IqlBiddingStrategy(),
                    MbrlComboMicroBiddingStrategy()
                ])
        return agents

    def reset(self, episode: int):
        """Reset the environment and agents for a new episode."""
        self.biddingEnv.reset(episode=episode)
        for agent in self.agents:
            agent.reset()
        if self.pv_generator_type == "modelPvGen":
            # Rotate categories each episode for diversity: cycle through all 44 categories
            start_cat = (episode * 6) % 44
            all_cats = np.arange(1, 45)
            select_category = list(all_cats[start_cat:start_cat + 6]) if start_cat + 6 <= 44 else \
                              list(all_cats[start_cat:]) + list(all_cats[:start_cat + 6 - 44])
            self.pvGenerator = ModelPvGenerator(
                num_tick=self.num_tick, num_agent_category=self.num_agent_category,
                select_category=select_category, pv_num=self.pv_num,
                episodic_std=0.1, episode=episode)
        else:
            self.pvGenerator.reset(episode=episode)

    def load_pv_generator(self):
        """Load the PV generator."""
        if self.pv_generator_type=="neuripsPvGen":
            return NeurIPSPvGen(episode=0,num_tick=self.num_tick, num_agent=self.num_agent,num_agent_category =self.num_agent_category, num_category=self.num_category,pv_num=self.pv_num)
        elif self.pv_generator_type=="modelPvGen":
            select_category = np.random.choice(np.arange(1, 45), size=6, replace=False)
            return ModelPvGenerator(num_tick=self.num_tick, num_agent_category=self.num_agent_category,
                                    select_category=select_category, pv_num=self.pv_num, episode=0)

    def load_bidding_env(self) -> BiddingEnv:
        """Load the bidding environment."""
        return BiddingEnv()

    def load_agents(self) -> list:
        """Load agents with their budget, CPA, and category."""
        for index in range(self.num_agent):
            self.agent_list[index].budget = self.budget_list[index]
            self.agent_list[index].cpa = self.cpa_constraint_list[index]
            self.agent_list[index].category = self.category[index]
            self.agent_list[index].name += str(index)
            self.agent_list[index].reset()

        self.player_agent.budget = self.budget_list[self.player_index]
        self.player_agent.cpa = self.cpa_constraint_list[self.player_index]
        self.player_agent.category = self.category[self.player_index]

        self.agent_list[self.player_index] = PlayerAgentWrapper(player_agent=self.player_agent)

        return self.agent_list

    def get_cpa_constraints(self) -> np.ndarray:
        """Get CPA constraints for the agents."""
        return np.array([
            100, 70, 90, 110, 60, 130, 120, 80,
            70, 130, 100, 110, 120, 90, 60, 80,
            130, 80, 110, 100, 90, 120, 60, 70,
            120, 60, 90, 70, 100, 110, 130, 80,
            120, 90, 70, 80, 100, 110, 60, 130,
            90, 100, 110, 80, 60, 70, 130, 120
        ])

    def calculate_budget(self) -> list:
        """Calculate the budget for each agent."""
        BUDGET_RATIO = 1
        budget = np.array([
            2900, 4350, 3000, 2400, 4800, 2000, 2050, 3500,
            4600, 2000, 2800, 2350, 2050, 2900, 4750, 3450,
            2000, 3500, 2200, 2700, 3100, 2100, 4850, 4100,
            2000, 4800, 3050, 4250, 2850, 2250, 2000, 3900,
            2000, 3250, 4450, 3550, 2700, 2100, 4650, 2000,
            3400, 2650, 2300, 4100, 4800, 4450, 2000, 2050
        ])
        return [x * BUDGET_RATIO for x in budget]


if __name__ == '__main__':
    os.chdir('../../')
    controller = Controller(player_agent=PidBiddingStrategy())
    cpa_constraints = np.array([1.1, 5.4, 0.9, 1.0, 0.9, 0.8, 1.6, 1.2, 1.4, 1.1, 1.0, 2.3, 1.2,
                                1.9, 0.8, 1.0, 2.2, 1.1, 1.0, 0.7, 1.0, 1.9, 0.8, 2.1, 1.5, 1.5,
                                1.8, 0.9, 1.1, 1.5])
    cpa_constraints *= 0.7
    tem = [round(x, 1) for x in cpa_constraints.tolist()]
    for i in range(30):
        print(f'{controller.agent_list[i].name} {controller.agent_list[i].cpa} {controller.agent_list[i].category}')
    print(tem)
