import gin
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


@gin.configurable
class PlayerAnalysis:
    """Class for analyzing player performance over multiple episodes."""

    def __init__(self, name: str = "PlayerAnalysis"):
        self.multi_episode_data: List[Tuple] = []
        self.analysis_res: List[Dict] = []

    def reset(self) -> None:
        """Reset the analysis data."""
        self.multi_episode_data.clear()
        self.analysis_res.clear()

    def logging_player_tick(self, episode: int, tick: int, player_index: int, cpa_constraint: float, budget: float,
                            tick_value: float, tick_cost: float, tick_compete_pv: int, tick_win_pv: int,
                            tick_all_win_bid: float, bid_mean: float) -> None:
        """Log data for each tick."""
        self.multi_episode_data.append(
            (player_index, episode, tick, cpa_constraint, budget, tick_value, tick_cost, tick_compete_pv,
             tick_win_pv, tick_all_win_bid, bid_mean)
        )

    def player_episode_index_analysis(self, df: pd.DataFrame, episode: int, player_index: int, name: str) -> Dict:
        df = df.sort_values('tick')
        analysis_result = self._analyze_player_episode(df)
        analysis_result["policyName"] = name
        analysis_result["episode"] = episode
        analysis_result["playerindex"] = player_index
        analysis_result["score"] = self._get_score_neurips(analysis_result["reward"], analysis_result["cpa"],
                                                        analysis_result["cpaConstraint"])
        return analysis_result

    def _analyze_player_episode(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze a single episode for a player."""
        all_value = np.sum(df["tickValue"])
        all_cost = np.sum(df["tickCost"])
        cpa_constraint = np.mean(df["cpa_constraint"])
        budget = np.mean(df["budget"])
        real_cpa = all_cost / (all_value + 1e-10)
        target_value = all_value * cpa_constraint
        all_compete_pv = np.sum(df["tickCompetePv"])
        all_win_pv = np.sum(df["tickWinPv"])
        all_win_bid = np.sum(df["tickAllWinBid"])
        budget_consumer_ratio = all_cost / budget
        second_price_ratio = all_cost / (all_win_bid + 1e-10)
        cpa_exceedance_rate = (real_cpa - cpa_constraint) / (cpa_constraint + 1e-10)
        bid_mean = np.mean(df["bidMean"])
        last_compete_tick_index = self._find_last_non_zero_index(df["tickWinPv"].tolist())
        win_pv_ratio = all_win_pv / (all_compete_pv + 1e-10)

        return {
            'cpaConstraint': cpa_constraint,
            'budget': budget,
            'reward': all_value,
            'allCost': all_cost,
            'cpa': real_cpa,
            'targetValue': target_value,
            'allCompetePv': all_compete_pv,
            'allWinPv': all_win_pv,
            'win_pv_ratio': win_pv_ratio,
            'budget_consumer_ratio': budget_consumer_ratio,
            'second_price_ratio': second_price_ratio,
            'cpa_exceedance_Rate': cpa_exceedance_rate,
            'last_compete_tick_index': last_compete_tick_index,
            'bidMean': bid_mean
        }

    @staticmethod
    def _find_last_non_zero_index(lst: List[int]) -> int:
        """Find the last non-zero index in a list."""
        for i in range(len(lst) - 1, -1, -1):
            if lst[i] != 0:
                return i
        return -1

    def player_multi_episode(self, policy_name: str) -> None:
        """Analyze multiple episodes for a player."""
        columns = ["playerindex", "episode", "tick", "cpa_constraint", "budget", "tickValue", "tickCost",
                   "tickCompetePv", "tickWinPv", "tickAllWinBid", "bidMean"]
        df = pd.DataFrame(self.multi_episode_data, columns=columns)
        grouped_data = df.groupby(['playerindex', 'episode'])

        for (player_index, episode), group in grouped_data:
            group = group.sort_values('tick')
            analysis_result = self.player_episode_index_analysis(group, episode, player_index, policy_name)
            self.analysis_res.append(analysis_result)

    def _get_score_neurips(self, reward: float, cpa: float, cpa_constraint: float) -> float:
        """Calculate the score using a neurips-competition penalty function."""
        beta = 2
        penalty = 1
        if cpa > cpa_constraint:
            coef = cpa_constraint / (cpa + 1e-10)
            penalty = pow(coef, beta)
        return penalty * reward

    def get_return_res(self, policy_name: str, player_index: int, category: str) -> Dict:
        """return the analysis results."""
        return_res = {
            "policyName": policy_name,
            "playerindex": player_index,
            "category": category,
            "rawData": self.analysis_res
        }

        for name in ['cpaConstraint', 'budget', 'reward', 'allCost', 'cpa', 'targetValue', 'allCompetePv', 'allWinPv',
                     'win_pv_ratio', 'budget_consumer_ratio', 'second_price_ratio', 'cpa_exceedance_Rate',
                     'last_compete_tick_index', 'bidMean']:
            tem_list = [x[name] for x in self.analysis_res]
            return_res[name] = np.sum(tem_list) if name == 'reward' else np.mean(tem_list)

        scores = [self._get_score_neurips(x["reward"], x["cpa"], x["cpaConstraint"]) for x in self.analysis_res]
        return_res["score"] = np.sum(scores) / 20000

        return return_res


def test():
    """Placeholder test function."""
    pass


if __name__ == '__main__':
    test()
