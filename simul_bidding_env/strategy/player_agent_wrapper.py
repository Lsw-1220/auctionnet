import time

from func_timeout import func_set_timeout, FunctionTimedOut


def custom_timeout(msg="action time out"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                # 尝试执行原始函数
                return func(*args, **kwargs)
            except FunctionTimedOut:
                self = args[0]
                if isinstance(self, PlayerAgentWrapper):
                    msg = f"agent {self.name} action time out"
                raise FunctionTimedOut(msg=msg)

        return wrapper

    return decorator


class PlayerAgentWrapper:
    def __init__(self, player_agent):
        super().__setattr__('player_agent', player_agent)

    def __setattr__(self, key, value):
        if key == 'player_agent':
            super().__setattr__(key, value)
        else:
            setattr(self.player_agent, key, value)

    def __getattr__(self, name):
        return getattr(self.player_agent, name)

    @func_set_timeout(timeout=2)
    def reset(self):
        self.player_agent.reset()

    @custom_timeout()
    @func_set_timeout(timeout=2)
    def action(self, tickIndex, budget, remaining_budget, pv_pvalues, HistoryPv,
               HistoryBid,
               HistoryStatus, HistoryReward, HistoryMarketprice):
        return self.player_agent.action(tickIndex, budget, remaining_budget, pv_pvalues, HistoryPv,
                                        HistoryBid,
                                        HistoryStatus, HistoryReward, HistoryMarketprice)
