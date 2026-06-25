import gin
import numpy as np
from scipy.stats import truncnorm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@gin.configurable
class BiddingEnv:
    """A bidding environment for simulating ad auctions."""

    def __init__(self, reserve_pv_price: float = 0.01, min_remaining_budget: float = 0.1):
        self.reserve_pv_price = reserve_pv_price
        self.min_remaining_budget = min_remaining_budget
        self.slot_coefficients = np.array([1, 0.8, 0.6])
        self.NUM_ADVERTISERS = 48
        self.advertiser_trunc_values = [(1, 0.01)] * self.NUM_ADVERTISERS
        self.MAGIC_NUMBER = 1019
        self.NUM_SLOTS = 3
        self.DEFAULT_SEED = 1

    def generate_trunc_values(self, advertiser_index: int, time_step_index: int, episode: int) -> tuple[
        int, float, float]:
        """Generates truncation values for a given advertiser and time step."""
        seed = hash((advertiser_index, time_step_index, episode)) + self.MAGIC_NUMBER
        seed = seed & ((1 << 32) - 1)
        rng = np.random.default_rng(seed)
        return seed, rng.random(), rng.random()

    def reset(self, episode: int) -> None:
        """Resets the environment for a new episode."""
        self.advertiser_trunc_values = [
            self.generate_trunc_values(advertiser_index, 0, episode)[1:]
            for advertiser_index in range(self.NUM_ADVERTISERS)
        ]

    def simulate_ad_bidding(self, pv_values: np.ndarray, p_value_sigmas: np.ndarray, bids: np.ndarray) -> tuple:
        """Simulates the ad bidding process."""
        xi, slot, cost = np.zeros_like(pv_values), np.zeros_like(pv_values), np.zeros_like(pv_values)

        sorted_bid_indices, market_prices = self._get_sorted_bids_and_market_prices(bids)
        slot, xi = self._assign_slots_and_xi(sorted_bid_indices, slot, xi)
        cost = self._calculate_cost(slot, market_prices)

        is_exposed = self._calculate_exposure(slot)
        values = self._generate_values_matrix(pv_values, p_value_sigmas)
        conversion_action = self._calculate_conversion_action(values, is_exposed)

        self._handle_unsold_slots(cost, xi, slot, is_exposed, conversion_action, market_prices)

        least_winning_cost = market_prices[:, -1]
        return xi.T, slot.T, cost.T, is_exposed.T, conversion_action.T, least_winning_cost, market_prices

    def _get_sorted_bids_and_market_prices(self, bids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sorts bids and calculates market prices."""
        sorted_bid_indices = np.argsort(-bids, axis=1)[:, :self.NUM_SLOTS]
        sorted_bids = -np.sort(-bids, axis=1)[:, :self.NUM_SLOTS + 1]
        market_prices = sorted_bids[:, 1:self.NUM_SLOTS + 1]
        market_prices[market_prices < self.reserve_pv_price] = self.reserve_pv_price
        return sorted_bid_indices, market_prices

    def _assign_slots_and_xi(self, sorted_bid_indices: np.ndarray, slot: np.ndarray, xi: np.ndarray) -> tuple[
        np.ndarray, np.ndarray]:
        """Assigns slots and xi values based on sorted bids."""
        values_to_put = np.arange(1, sorted_bid_indices.shape[1] + 1)
        np.put_along_axis(slot, sorted_bid_indices, values_to_put[None, :], axis=1)
        slot = slot.astype(int)
        xi = (slot > 0).astype(int)
        return slot, xi

    def _calculate_cost(self, slot: np.ndarray, market_prices: np.ndarray) -> np.ndarray:
        """Calculates the cost for each slot."""
        rows_indices = np.indices(slot.shape)[0]
        column_indices = slot - 1
        result = market_prices[rows_indices, column_indices]
        result[slot == 0] = 0
        return result

    def _calculate_exposure(self, slot: np.ndarray) -> np.ndarray:
        """Calculates exposure values for each slot."""
        is_exposed = self.slot_coefficients[slot - 1]
        is_exposed[slot == 0] = 0
        rng = np.random.default_rng(seed=self.DEFAULT_SEED)
        is_exposed = rng.binomial(n=1, p=np.clip(is_exposed, 0, 1))
        return self._enforce_slot_continuity(is_exposed, slot)

    def _enforce_slot_continuity(self, is_exposed: np.ndarray, slot: np.ndarray) -> np.ndarray:
        """Ensures continuity rules for slot exposure."""
        is_exposed_0_for_slot_2 = ((slot == 2) & (is_exposed == 0))
        slot_3_positions = slot == 3
        update_positions = (is_exposed_0_for_slot_2.any(axis=1).reshape(-1, 1) & slot_3_positions)
        is_exposed[update_positions] = 0
        return is_exposed

    def _generate_values_matrix(self, pv_values: np.ndarray, p_value_sigmas: np.ndarray) -> np.ndarray:
        """Generates a matrix of values for each advertiser."""
        num_pv, num_advertisers = pv_values.shape
        value1 = np.array([v[0] for v in self.advertiser_trunc_values]).reshape(1, num_advertisers)
        value2 = np.array([v[1] for v in self.advertiser_trunc_values]).reshape(1, num_advertisers)
        a_standardized = -2 * value1
        b_standardized = 2 * value2
        rng = np.random.default_rng(seed=self.DEFAULT_SEED)
        samples = truncnorm.rvs(a_standardized, b_standardized, loc=pv_values, scale=p_value_sigmas, random_state=rng)
        return samples

    def _calculate_conversion_action(self, values: np.ndarray, is_exposed: np.ndarray) -> np.ndarray:
        """Calculates conversion actions based on values and exposure."""
        rng = np.random.default_rng(seed=self.DEFAULT_SEED)
        conversion_action = rng.binomial(n=1, p=np.clip(values, 0, 1))
        return conversion_action * is_exposed

    def _handle_unsold_slots(self, cost: np.ndarray, xi: np.ndarray, slot: np.ndarray, is_exposed: np.ndarray,
                             conversion_action: np.ndarray, market_prices: np.ndarray) -> None:
        """Handles unsold slots by updating related arrays."""
        is_unsold = (cost == self.reserve_pv_price)
        xi[is_unsold] = 0
        slot[is_unsold] = 0
        cost[is_unsold] = 0
        is_exposed[is_unsold] = 0
        conversion_action[is_unsold] = 0


if __name__ == '__main__':
    logging.info("Bidding environment simulation started.")
    # Here you can add any test or example usage of the BiddingEnv class.
