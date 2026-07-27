"""
evaluate.py — 统一评估入口.

使用模式:
  1. DGAB PO 系列 (R0-R5, POMDP): strategy_type=dgab_po (default)
     python -m run.evaluate --config r0,r1,r4
     python -m run.evaluate --save_dir saved_model/DGAB_r1 --use_stat --actor_type stack

  2. DGAB Ensemble (R2+R5 稀疏度自适应): strategy_type=dgab_ensemble
     python -m run.evaluate --config r2r5

  3. DGAB FO (完全可观测对照): strategy_type=dgab_fo
     python -m run.evaluate --strategy_type dgab_fo --save_dir saved_model/DGAB_fo_xxx

  4. 旧 GAVE baseline (FO, state=16): strategy_type=gave
     python -m run.evaluate --strategy_type gave --save_dir saved_model/GAVE

  5. 旧 PlayerBiddingStrategy (DT/VCRTG 等): strategy_type=player
     python -m run.evaluate --strategy_type player --save_dir saved_model/DTtest_xxx
     python -m run.evaluate --strategy_type player --save_dir ... --scan_checkpoints

作为库函数调用:
  from run.evaluate import run_test
  score, score1, conversion, exceed = run_test(file_path, model_param, model_name=None)

  model_param['strategy_type'] 控制路径:
    'dgab_po'  (default) → DGABBiddingStrategy   (R0-R5, POMDP)
    'dgab_fo'            → DGABFOBiddingStrategy  (FO 变体)
    'gave'               → GAVEBiddingStrategy    (state=16, next/prev)
    'player'             → PlayerBiddingStrategy  (旧 DT/VCRTG)
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import glob
import logging
import argparse
from datetime import datetime
import numpy as np
import pandas as pd

from bidding_train_env.dataloader.test_dataloader import TestDataLoader
from bidding_train_env.environment.offline_env import OfflineEnv

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

BLOCK_CONFIG = {
    'n_ctx': 1024, 'n_embd': 512, 'n_layer': 8, 'n_head': 16,
    'n_inner': 1024, 'activation_function': 'relu', 'n_position': 1024,
    'resid_pdrop': 0.1, 'attn_pdrop': 0.1,
}

# 预设消融配置
MODEL_CONFIGS = {
    'r0': dict(save_dir='saved_model/dgab_po_r0', use_stat=False,
               actor_type='stack',      obs_encoder_type='none'),
    'r1': dict(save_dir='saved_model/dgab_po_r1', use_stat=True,
               actor_type='stack',      obs_encoder_type='none'),
    'r2': dict(save_dir='saved_model/dgab_po_r2', use_stat=False,
               actor_type='stack',      obs_encoder_type='cls'),
    'r3': dict(save_dir='saved_model/dgab_po_r3', use_stat=False,
               actor_type='stack',      obs_encoder_type='bidformer'),
    'r4': dict(save_dir='saved_model/dgab_po_r4', use_stat=True,
               actor_type='cross_attn', obs_encoder_type='none'),
    'r5': dict(save_dir='saved_model/dgab_po_r5', use_stat=True,
               actor_type='cross_attn', obs_encoder_type='bidformer'),
    # R6: BidFormer + CrossAttnActor without stat_t (isolates stat_t effect)
    'r6': dict(save_dir='saved_model/dgab_po_r6', use_stat=False,
               actor_type='cross_attn', obs_encoder_type='bidformer'),
    # R7: stat_t in memory only, excluded from V/C query (attenuated density signal)
    'r7': dict(save_dir='saved_model/dgab_po_r7', use_stat=True,
               actor_type='cross_attn_r7', obs_encoder_type='bidformer'),
    # Ensemble: R2 (dense, cls-token) + R5 (sparse, BidFormer)
    # sparsity_threshold: z-scored log(n_imp+1) below which → R5
    'r2r5': dict(
        strategy_type='dgab_ensemble',
        dense_model=dict(
            save_dir='saved_model/dgab_po_r2',
            use_stat=False, actor_type='stack', obs_encoder_type='cls',
        ),
        sparse_model=dict(
            save_dir='saved_model/dgab_po_r5',
            use_stat=True, actor_type='cross_attn', obs_encoder_type='bidformer',
        ),
        sparsity_threshold=0.0,
    ),
    # Fusion: Plan B — dual-encoder with heuristic gate (requires trained DGABFusion)
    'fusion': dict(save_dir='saved_model/DGAB_fusion',
                   strategy_type='dgab_fusion'),
}


def getScore_nips(reward, cpa, cpa_constraint):
    beta = 2
    penalty = 1.0
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward


def getScore1_nips(reward, cpa, cpa_constraint):
    beta = 5
    penalty = 1.0
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward


def _make_agent(budget, cpa, category, model_param, model_name=None):
    """Instantiate the right agent depending on strategy_type.

    strategy_type values:
      'dgab_po'  (default) → DGABBiddingStrategy    (R0-R5, POMDP)
      'dgab_fo'            → DGABFOBiddingStrategy   (FO 变体)
      'gave'               → GAVEBiddingStrategy     (state=16, next/prev)
      'player'             → PlayerBiddingStrategy   (legacy DT)
    """
    strategy_type = model_param.get('strategy_type', 'dgab_po')
    if strategy_type == 'player':
        from bidding_train_env.strategy import PlayerBiddingStrategy
        return PlayerBiddingStrategy(
            budget=budget, cpa=cpa, category=category,
            model_name=model_name or 'complete_train.pt',
            model_param=model_param)
    elif strategy_type == 'gave':
        from bidding_train_env.strategy.GAVE_bidding_strategy import GAVEBiddingStrategy
        return GAVEBiddingStrategy(
            budget=budget, cpa=cpa, category=category, model_param=model_param)
    elif strategy_type == 'dgab_fo':
        from bidding_train_env.strategy.dgab_fo_bidding_strategy import DGABFOBiddingStrategy
        return DGABFOBiddingStrategy(
            budget=budget, cpa=cpa, category=category, model_param=model_param)
    elif strategy_type == 'dgab_ensemble':
        from bidding_train_env.strategy.dgab_ensemble_strategy import (
            DGABEnsembleBiddingStrategy)
        return DGABEnsembleBiddingStrategy(
            budget=budget, cpa=cpa, category=category, model_param=model_param)
    elif strategy_type == 'dgab_fusion':
        from bidding_train_env.strategy.dgab_fusion_strategy import (
            DGABFusionBiddingStrategy)
        return DGABFusionBiddingStrategy(
            budget=budget, cpa=cpa, category=category, model_param=model_param)
    else:  # dgab_po
        from bidding_train_env.strategy.dgab_bidding_strategy import (
            DGABBiddingStrategy)
        mp = dict(model_param)
        if model_name is not None:
            mp['_ckpt_name'] = model_name
        return DGABBiddingStrategy(
            budget=budget, cpa=cpa, category=category, model_param=mp)


def run_test(file_path, model_param, model_name=None, tick_collector=None):
    """
    Unified test runner.
    Returns (score, score1, conversion, exceed_rate).

    tick_collector: optional list — if provided, per-tick dicts are appended to it.
      Each dict has keys: period, advertiser, tick, n_opp, won, win_rate,
        avg_bid, avg_pv, avg_lwc, tick_conv, tick_cost, cum_conv, cum_cost,
        cum_cpa, remaining_budget.
    """
    data_loader = TestDataLoader(file_path=file_path)
    env = OfflineEnv()
    keys = data_loader.keys

    overall_score  = 0.0
    overall_score1 = 0.0
    overall_conv   = 0.0
    exceed_count   = 0

    for key in keys:
        (num_timeStepIndex, pValues, pValueSigmas,
         leastWinningCosts, budget, cpa, category) = data_loader.mock_data(key)
        budget = budget * model_param.get('budget_rate', 1.0)

        agent = _make_agent(budget, cpa, category, model_param, model_name)

        rewards = np.zeros(num_timeStepIndex)
        history = {
            'historyBids': [],
            'historyAuctionResult': [],
            'historyImpressionResult': [],
            'historyLeastWinningCost': [],
            'historyPValueInfo': [],
        }

        for t in range(num_timeStepIndex):
            pValue           = pValues[t]
            pValueSigma      = pValueSigmas[t]
            leastWinningCost = leastWinningCosts[t]

            if agent.remaining_budget < env.min_remaining_budget:
                bid = np.zeros(pValue.shape[0])
            else:
                bid = agent.bidding(
                    t, pValue, pValueSigma,
                    history['historyPValueInfo'], history['historyBids'],
                    history['historyAuctionResult'], history['historyImpressionResult'],
                    history['historyLeastWinningCost'])

            tick_value, tick_cost, tick_status, tick_conversion = \
                env.simulate_ad_bidding(pValue, pValueSigma, bid, leastWinningCost)

            over_cost_ratio = max(
                (np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
            while over_cost_ratio > 0:
                pv_index = np.where(tick_status == 1)[0]
                n_drop   = min(int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                               pv_index.shape[0])
                bid[np.random.choice(pv_index, n_drop, replace=False)] = 0
                tick_value, tick_cost, tick_status, tick_conversion = \
                    env.simulate_ad_bidding(pValue, pValueSigma, bid, leastWinningCost)
                over_cost_ratio = max(
                    (np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)

            agent.remaining_budget -= np.sum(tick_cost)
            rewards[t] = np.sum(tick_conversion)

            cum_conv = float(np.sum(rewards[:t+1]))
            cum_cost = float(agent.budget - agent.remaining_budget)
            if tick_collector is not None:
                n_opp = int(pValue.shape[0])
                won   = int((tick_status == 1).sum())
                tick_collector.append(dict(
                    period=key[0], advertiser=key[1], tick=t,
                    n_opp=n_opp, won=won,
                    win_rate=won / (n_opp + 1e-10),
                    avg_bid=float(np.mean(bid))          if n_opp > 0 else 0.0,
                    avg_pv=float(np.mean(pValue))        if n_opp > 0 else 0.0,
                    avg_lwc=float(np.mean(leastWinningCost)) if n_opp > 0 else 0.0,
                    tick_conv=float(np.sum(tick_conversion)),
                    tick_cost=float(np.sum(tick_cost)),
                    cum_conv=cum_conv, cum_cost=cum_cost,
                    cum_cpa=cum_cost / (cum_conv + 1e-10),
                    remaining_budget=float(agent.remaining_budget),
                ))

            history['historyPValueInfo'].append(
                np.array([(pValue[i], pValueSigma[i]) for i in range(pValue.shape[0])]))
            history['historyBids'].append(bid)
            history['historyLeastWinningCost'].append(leastWinningCost)
            history['historyAuctionResult'].append(np.array([
                (pValue[i], pValueSigma[i], tick_status[i], 0.0,
                 tick_cost[i], tick_status[i], tick_conversion[i])
                for i in range(tick_status.shape[0])]))
            history['historyImpressionResult'].append(
                np.array([(tick_conversion[i], tick_conversion[i])
                          for i in range(pValue.shape[0])]))

        all_reward = np.sum(rewards)
        all_cost   = agent.budget - agent.remaining_budget
        cpa_real   = all_cost / (all_reward + 1e-10)

        overall_score  += getScore_nips(all_reward,  cpa_real, agent.cpa)
        overall_score1 += getScore1_nips(all_reward, cpa_real, agent.cpa)
        overall_conv   += all_reward
        if cpa_real > agent.cpa:
            exceed_count += 1

        logger.info(f"reward={all_reward:.2f} cost={all_cost:.2f} "
                    f"cpa_real={cpa_real:.4f} cpa_target={agent.cpa:.4f}")

    n = len(keys)
    return overall_score / n, overall_score1 / n, overall_conv / n, exceed_count / n


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy_type', type=str, default='dgab_po',
                        choices=['dgab_po', 'dgab_fo', 'gave', 'player'])
    parser.add_argument('--config', type=str, default='',
                        help="gave模式: 逗号分隔的消融名 r0,r1,r2,r3,r4,r5")
    parser.add_argument('--data',   type=str, default='data/MDP/traffic/period-7.csv')
    parser.add_argument('--data_dir', type=str, default='',
                        help='测试文件夹路径，测试该文件夹下所有 period-*.csv')
    parser.add_argument('--output_dir', type=str, default='log',
                        help='结果 CSV 保存目录 (default: log/)')
    parser.add_argument('--budget_rate', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cpu')
    # single-model overrides (gave模式)
    parser.add_argument('--save_dir',        type=str, default='')
    parser.add_argument('--save_dirs',       type=str, nargs='*', default=[],
                        help='多个模型目录，自动从 normalize_dict.pkl 读取配置')
    parser.add_argument('--use_stat',        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--actor_type',      type=str, default='stack',
                        choices=['stack', 'cross_attn'])
    parser.add_argument('--obs_encoder_type', type=str, default='none',
                        choices=['none', 'cls', 'bidformer'])
    # scan mode (player模式 / 旧行为)
    parser.add_argument('--scan_checkpoints', action='store_true',
                        help="扫描 save_dir 下所有 .pt 文件逐一评估 (player 模式默认行为)")
    # transformer hyperparams (player模式)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--time_dim',    type=int, default=8)
    parser.add_argument('--n_ctx',    type=int, default=1024)
    parser.add_argument('--n_embd',   type=int, default=512)
    parser.add_argument('--n_layer',  type=int, default=8)
    parser.add_argument('--n_head',   type=int, default=16)
    parser.add_argument('--n_inner',  type=int, default=1024)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--n_position', type=int, default=1024)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop',  type=float, default=0.1)
    parser.add_argument('--state_dim', type=int, default=16)
    parser.add_argument('--act_dim',   type=int, default=1)
    args = parser.parse_args()

    block_config = {
        'n_ctx': args.n_ctx, 'n_embd': args.n_embd, 'n_layer': args.n_layer,
        'n_head': args.n_head, 'n_inner': args.n_inner,
        'activation_function': args.activation_function,
        'n_position': args.n_position,
        'resid_pdrop': args.resid_pdrop, 'attn_pdrop': args.attn_pdrop,
    }

    # ── 决定要跑哪些 config ──
    if args.strategy_type == 'dgab_fo':

        if not args.save_dir:
            print("ERROR: --save_dir is required for dgab_fo")
            exit(1)
        mp = dict(
            strategy_type='dgab_fo',
            save_dir=args.save_dir,
            hidden_size=args.hidden_size,
            max_ep_len=96,
            time_dim=args.time_dim,
            block_config=block_config,
            device=args.device,
            budget_rate=args.budget_rate,
            K=20,
        )
        traffic_csvs = sorted(glob.glob('data/MDP/traffic/period-*.csv'))
        if not traffic_csvs:
            traffic_csvs = [args.data]
        results = []
        for csv_path in traffic_csvs:
            period_tag = os.path.splitext(os.path.basename(csv_path))[0]
            score, score1, conv, exc = run_test(csv_path, mp)
            results.append(dict(period=period_tag, score=score, conversion=conv, exceed_rate=exc))
            print(f"  {period_tag}  score={score:.4f}  conversion={conv:.2f}  exceed={exc:.2%}")
        print('\n── Summary ──')
        print(f"{'Period':<12} {'Score':>10} {'Conversion':>12} {'ExceedRate':>12}")
        for r in results:
            print(f"{r['period']:<12} {r['score']:>10.4f} "
                  f"{r['conversion']:>12.2f} {r['exceed_rate']:>11.2%}")

    elif args.strategy_type == 'dgab_po':
        if args.save_dirs:
            # 从 normalize_dict.pkl 自动检测配置
            import pickle
            configs_to_run = {}
            for sd in args.save_dirs:
                name = os.path.basename(sd.rstrip('/\\'))
                nd_path = os.path.join(sd, 'normalize_dict.pkl')
                if not os.path.isfile(nd_path):
                    print(f'[SKIP] {sd}: normalize_dict.pkl not found')
                    continue
                nd = pickle.load(open(nd_path, 'rb'))
                # base_state_dim=18 表示 use_stat=True, =2 表示 False
                use_stat = nd.get('base_state_dim', 2) >= 18
                configs_to_run[name] = dict(
                    save_dir=sd,
                    use_stat=use_stat,
                    actor_type=nd.get('actor_type', 'stack'),
                    obs_encoder_type=nd.get('obs_encoder_type', 'none'))
        elif args.save_dir:
            configs_to_run = {'custom': dict(
                save_dir=args.save_dir, use_stat=args.use_stat,
                actor_type=args.actor_type, obs_encoder_type=args.obs_encoder_type)}
        else:
            names = [c.strip().lower() for c in args.config.split(',') if c.strip()]
            if not names:
                names = ['r0', 'r1', 'r4']
            configs_to_run = {k: MODEL_CONFIGS[k] for k in names if k in MODEL_CONFIGS}

        # 确定测试文件列表
        if args.data_dir:
            test_files = sorted(glob.glob(os.path.join(args.data_dir, 'period-*.csv')))
            if not test_files:
                print(f'[ERROR] No period-*.csv files found in {args.data_dir}')
                exit(1)
        else:
            test_files = [args.data]

        all_results = []
        for csv_path in test_files:
            period_tag = os.path.splitext(os.path.basename(csv_path))[0]
            print(f'\n{"="*50}')
            print(f'Period: {period_tag}')
            print(f'{"="*50}')
            for name, cfg in configs_to_run.items():
                print(f'  Evaluating {name.upper()} ... ', end='', flush=True)
                # Config may specify its own strategy_type (e.g. ensemble), default to dgab_po
                mp = dict(hidden_size=args.hidden_size, max_ep_len=96, time_dim=args.time_dim,
                          block_config=BLOCK_CONFIG, device=args.device,
                          budget_rate=args.budget_rate, K=20,
                          strategy_type='dgab_po', **cfg)
                score, score1, conv, exc = run_test(csv_path, mp)
                all_results.append(dict(period=period_tag, config=name,
                                        score=score, conversion=conv, exceed_rate=exc))
                print(f'score={score:.4f}  conversion={conv:.2f}  exceed={exc:.2%}')

        if len(all_results) > 1:
            print('\n── Summary ──')
            print(f"{'Period':<12} {'Config':<8} {'Score':>10} {'Conversion':>12} {'ExceedRate':>12}")
            print('-' * 62)
            for r in all_results:
                print(f"{r['period']:<12} {r['config'].upper():<8} {r['score']:>10.4f} "
                      f"{r['conversion']:>12.2f} {r['exceed_rate']:>11.2%}")

        # 保存结果到 output_dir
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(args.output_dir, exist_ok=True)
        out_csv = os.path.join(args.output_dir, f"eval_dgab_po_{ts}.csv")
        pd.DataFrame(all_results,
                     columns=['period', 'config', 'score', 'conversion', 'exceed_rate']
                     ).to_csv(out_csv, index=False)
        print(f'Results saved to {out_csv}')

    else:
        # player 模式 — 兼容旧 run_evaluate_ori 行为
        base_mp = dict(
            hidden_size=args.hidden_size, time_dim=args.time_dim,
            budget_rate=args.budget_rate, device=args.device,
            save_dir=args.save_dir, block_config=block_config,
            state_dim=args.state_dim, act_dim=args.act_dim,
            strategy_type='player',
        )

        pt_files = glob.glob(os.path.join(args.save_dir, '*.pt'))
        pt_stems = [os.path.splitext(os.path.basename(p))[0] for p in pt_files]
        numeric     = sorted((s for s in pt_stems if s.isdigit()), key=int)
        non_numeric = sorted(s for s in pt_stems if not s.isdigit())
        pt_names    = numeric + non_numeric

        if not args.scan_checkpoints or not pt_names:
            pt_names = ['complete_train']

        eval_result = []
        for pt_name in pt_names:
            score, score1, conv, exc = run_test(
                args.data, base_mp, model_name=f'{pt_name}.pt')
            eval_result.append([pt_name, score, score1, conv, exc])
            print(f"  {pt_name}.pt  score={score:.4f}  exceed={exc:.2%}")

        out_csv = os.path.join(args.save_dir, 'eval_result.csv')
        pd.DataFrame(eval_result,
                     columns=['file', 'score', 'score1', 'conversion', 'exceed']
                     ).sort_values('file').to_csv(out_csv, index=False)
        print(f"Results saved to {out_csv}")
