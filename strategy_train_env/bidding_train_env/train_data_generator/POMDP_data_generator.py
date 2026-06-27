import glob, os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class POMDPDataGenerator:
    """
    Read raw auction log CSV and construct POMDP trajectory data.
    """

    def __init__(self):
        self.timeStepIndexNum = 48
        self.observation_list = ['pValue', 'pValueSigma', 'xi', 'adSlot', 'cost',
                                 'isExposed', 'conversionAction', 'bid', 'leastWinningCost']

    def convert_dataframe(self, df):
        """Convert a raw auction DataFrame to POMDP format (in-memory)."""
        training_data_rows = []

        for (period, adv, cat, budget, cpa), group in df.groupby(
                ['deliveryPeriodIndex', 'advertiserNumber',
                 'advertiserCategoryIndex', 'budget', 'CPAConstraint']):

            group = group.sort_values('timeStepIndex')

            for ts in group['timeStepIndex'].unique():
                cur = group[group['timeStepIndex'] == ts]

                timeleft = (self.timeStepIndexNum - ts) / self.timeStepIndexNum
                remaining = cur['remainingBudget'].iloc[0]
                bgtleft = remaining / budget if remaining > 0 else 0

                if ts == 0:
                    obs = np.zeros([1, 9], dtype=np.float32)
                else:
                    obs = group[group['timeStepIndex'] == ts - 1][self.observation_list].values.astype(np.float32)

                total_bid = cur['bid'].sum()
                total_value = cur['pValue'].sum()
                action = total_bid / total_value if total_value > 0 else 0

                reward = cur[cur['isExposed'] == 1]['conversionAction'].sum()
                reward_cont = cur[cur['isExposed'] == 1]['pValue'].sum()

                done = 1 if ts == self.timeStepIndexNum - 1 or cur['isEnd'].iloc[0] == 1 else 0

                training_data_rows.append({
                    'deliveryPeriodIndex': period,
                    'advertiserNumber': adv,
                    'advertiserCategoryIndex': cat,
                    'budget': budget,
                    'CPAConstraint': cpa,
                    'timeStepIndex': ts,
                    'resourceLeft': (timeleft, bgtleft),
                    'observations': obs,
                    'action': action,
                    'reward': reward,
                    'reward_continous': reward_cont,
                    'done': done,
                })

        return pd.DataFrame(training_data_rows)

    def convert_csv_to_pickle(self, csv_path, pkl_dir):
        """Read a CSV, convert to POMDP, save as pickle, return df."""
        df = pd.read_csv(csv_path)
        df_pomdp = self.convert_dataframe(df)
        os.makedirs(pkl_dir, exist_ok=True)
        name = os.path.basename(csv_path).replace('.csv', '-POMDP.pkl')
        pkl_path = os.path.join(pkl_dir, name)
        df_pomdp.to_pickle(pkl_path)
        return df_pomdp, pkl_path

    def batch_convert(self, csv_dir, pkl_dir):
        """Convert all CSVs in csv_dir to POMDP pickle files."""
        os.makedirs(pkl_dir, exist_ok=True)
        csv_files = sorted(glob.glob(os.path.join(csv_dir, '*.csv')))
        all_data = []
        for f in csv_files:
            df, pkl = self.convert_csv_to_pickle(f, pkl_dir)
            all_data.append(df)
            print(f"  Converted: {os.path.basename(f)} -> {os.path.basename(pkl)}")
        # Merge all
        merged = pd.concat(all_data, ignore_index=True)
        merged_path = os.path.join(pkl_dir, 'POMDP_all.pkl')
        merged.to_pickle(merged_path)
        print(f"  Merged: {merged_path} ({len(merged)} rows)")
        return merged
