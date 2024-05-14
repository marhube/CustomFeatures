import os
import sys # For testing
import re

from itertools import product
import datetime as dt
#Andre "built-in"-moduler

import pandas as pd
import numpy as np
from pandas.core.groupby import DataFrameGroupBy
from typing import List, Callable, Any
from scipy.stats import binom, t # type: ignore
#

class CumulativeMeanCalculator:
    def __init__(self, grouping_vars: list[str], numeric_col: str, time_var: str, prefix: str ='cum',signif_level:float = 0.05, dist: str='t',id_col=None):
        self.grouping_vars = grouping_vars
        self.numeric_col = numeric_col
        self.time_var = time_var
        self.prefix = prefix  # Store the prefix
        self.signif_level = signif_level # Store level of significance
        permitted_distributions = ["t","binom"]
        if not dist in permitted_distributions:
            raise ValueError(f": 'dist' should be one of the of the following: {permitted_distributions}")
        #
        self.dist = dist
        self.id_col = id_col
        
        
    def calculate_cumulative_means(self, df: pd.DataFrame) -> pd.DataFrame:
        required_columns = self.grouping_vars + [self.numeric_col, self.time_var]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame is missing the following required columns: {missing_columns}")

        df_sorted = df.sort_values(by=[self.time_var] + self.grouping_vars).reset_index(drop=True)
        summary_df = pd.DataFrame()
        time_values = df_sorted[self.time_var].unique()

        for t in time_values:
            temp_df = df_sorted[df_sorted[self.time_var] < t]
            for i in range(1, len(self.grouping_vars) + 1):
                current_group_vars = self.grouping_vars[:i]
                if temp_df.empty:
                    placeholder_row = self.create_placeholder_row(current_group_vars, t, i)
                    summary_df = pd.concat([summary_df, placeholder_row], ignore_index=True)
                else:
                    cum_stats = self.compute_cumulative_stats(temp_df, current_group_vars)
                    cum_stats[self.time_var] = t
                    cum_stats['grouping_level'] = i
                    summary_df = pd.concat([summary_df, cum_stats], ignore_index=True)

        return summary_df.drop_duplicates(subset=[self.time_var] + self.grouping_vars).sort_values(by=[self.time_var] + self.grouping_vars)

    def create_placeholder_row(self, grouping_vars: List[str],t: Any, level: int) -> pd.DataFrame:
        data = {var: np.nan for var in grouping_vars}
        data.update({
            self.time_var: t,
            f"{self.prefix}_mean": np.nan,
            f"{self.prefix}_sum": 0,
            f"{self.prefix}_count": 0,
            f"{self.prefix}_std_dev": np.nan,
            'grouping_level': level
        })
        return pd.DataFrame([data])


    def compute_cumulative_stats(self, df: pd.DataFrame, grouping_vars:List[str]):
        # Ensure 'self.time_var' is included in the grouping if not part of the original grouping_vars
        if self.time_var not in grouping_vars:
            grouping_vars_with_time = grouping_vars + [self.time_var]
        else:
            grouping_vars_with_time = grouping_vars
       # Aggregate using default names
        grouped = df.groupby(grouping_vars_with_time)
        aggregated = grouped.agg({
            self.numeric_col: ['sum', 'count', 'mean', lambda x: (x**2).sum()]
        }).reset_index()

        # Rename columns based on their content
        new_columns = []
        print(f"aggregated.columns is {aggregated.columns}")
        for col in aggregated.columns:
            print(f"col is {col}")
            if isinstance(col, tuple):
                if col[1] == '':  # This means it's actually a grouping variable
                    new_columns.append(col[0])
                else:
                    print("Entering the if")
                    operation = col[1]
                    print(f"operation is {operation}")
                    if re.search(r"<lambda.*>",operation):
                        operation = "squares_sum"  # handling lambda function naming
                    new_name = f"{self.prefix}_{operation}"
                    new_columns.append(new_name)
            else:
                new_columns.append(col)  # Any column not in tuple form is passed directly

        # Convert list to pandas Index before assignment
        aggregated.columns = pd.Index(new_columns)

        #
        print(f"list(aggregated.columns) is now{list(aggregated.columns)}")


        # Calculate cumulative variance and standard deviation
        aggregated[f"{self.prefix}_variance"] = (aggregated[f"{self.prefix}_squares_sum"] / aggregated[f"{self.prefix}_count"]) - (aggregated[f"{self.prefix}_mean"] ** 2)
        aggregated[f"{self.prefix}_std_dev"] = np.sqrt(aggregated[f"{self.prefix}_variance"])

        # Set variance and standard deviation to NaN where cum_count is 1 or less
        aggregated.loc[aggregated[f"{self.prefix}_count"] <= 1, f"{self.prefix}_variance"] = np.nan
        aggregated.loc[aggregated[f"{self.prefix}_count"] <= 1, f"{self.prefix}_std_dev"] = np.nan

        # Include the grouping level information
        aggregated['grouping_level'] = len(grouping_vars)

        # Specify return columns including 'self.time_var'
        return_columns = grouping_vars + [self.time_var, f"{self.prefix}_mean", f"{self.prefix}_sum", f"{self.prefix}_count", f"{self.prefix}_squares_sum", f"{self.prefix}_variance", f"{self.prefix}_std_dev", 'grouping_level']
        return aggregated[return_columns]


    
    def prepare_level_data(self, df: pd.DataFrame, level: int) -> tuple[pd.DataFrame,list]:
        """ Prepare data for a specific grouping level with appropriate renaming. """
        # Filter data for the current level
        level_data = df[df['grouping_level'] == level]
        
        # Check which of the grouping variables are present in the data for this level
        available_group_vars = [var for var in self.grouping_vars[:level] if var in level_data.columns]
        print(f"Level {level} available variables: {available_group_vars}")  # Debugging output
        
        # Relevant statistics to keep
        stats_columns = [
            f"{self.prefix}_mean", f"{self.prefix}_sum", f"{self.prefix}_count", 
            f"{self.prefix}_std_dev", f"{self.prefix}_squares_sum", f"{self.prefix}_variance"
        ]
        
        # Select only the necessary columns, handling missing columns if they are not in the DataFrame
        selected_columns = [self.time_var] + available_group_vars + stats_columns
        selected_columns = [col for col in selected_columns if col in level_data.columns]
        level_data = level_data[selected_columns]


        # Rename statistics columns to include the level information
        renamed_columns = {col: f"{col}_level{level}" for col in stats_columns}
        level_data.rename(columns=renamed_columns, inplace=True)
        
        return level_data, available_group_vars

    def reshape_results(self, df: pd.DataFrame) -> pd.DataFrame:
        final_df = None

        # Process each level of granularity
        for level in range(1, len(self.grouping_vars) + 1):
            level_data, merge_cols = self.prepare_level_data(df, level)
            merge_cols = [self.time_var] + merge_cols  # Include the time variable in merge columns

            # Merge with the final DataFrame
            if final_df is None:
                final_df = level_data
            else:
                #Remove the last "merging column" that is not yet found in final_df
                merge_on_cols = [col for col in merge_cols if col in final_df.columns]
                final_df = pd.merge(final_df, level_data, on=merge_on_cols, how='left')
        
        #"Error handling" to keep mypy happy.
        if not isinstance(final_df,pd.DataFrame):
            raise ValueError("final_df is not a datatrame.")

        return final_df



    # fill_prediction_data Now modifies the table "in-place"    
    def fill_prediction_data(self, my_summation_table: pd.DataFrame, my_prediction_data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill in the prediction data with statistics from all levels of granularity,
        modifying the input DataFrame directly.

        Parameters:
            my_summation_table (pd.DataFrame): DataFrame containing cumulative statistics.
            my_prediction_data (pd.DataFrame): DataFrame to be updated with the latest available statistics.

        Returns:
            pd.DataFrame: Updated prediction data with new statistics columns.
        """
        # Ensure prediction data has necessary columns
        if not set(self.grouping_vars + [self.time_var]).issubset(my_prediction_data.columns):
            raise ValueError("Prediction data missing necessary columns.")

        # Sort summation table by time in descending order to prioritize latest data on merge
        my_summation_table_sorted = my_summation_table.sort_values(by=self.time_var, ascending=False)

        # Define stats columns
        stats = ['sum', 'count', 'mean', 'std_dev']

        # Merge data for each level of granularity
        for level in range(1, len(self.grouping_vars) + 1):
            current_groups = self.grouping_vars[:level]
            stat_cols = [f"{self.prefix}_{stat}_level{level}" for stat in stats]

            # Filter latest stats for current level and merge
            # Ensure to i nclude the time variable in merging keys to avoid suffix addition
            merge_keys = current_groups + [self.time_var]
            latest_stats = my_summation_table_sorted[merge_keys + stat_cols].drop_duplicates(subset=current_groups, keep='first')
            my_prediction_data = my_prediction_data.merge(latest_stats, on=merge_keys, how='left')

            # Ensure missing statistical values are handled appropriately
            for stat_col in stat_cols:
                if 'sum' in stat_col or 'count' in stat_col:
                    my_prediction_data[stat_col] = my_prediction_data[stat_col].fillna(0)
                else:
                    my_prediction_data[stat_col] = my_prediction_data[stat_col].fillna(np.nan)

        # Remove any unwanted columns that might have been introduced
        expected_cols = [self.time_var] + self.grouping_vars + list(my_prediction_data.columns[-len(stats) * len(self.grouping_vars):])
        return my_prediction_data[expected_cols]


    def tztest(self, mu0: float, mu1: float, n0: int, n1: int, sigma: float) -> float:
        if n0 == 0:
            return 0
        if n0 > 0 and n1 == 0:
            return mu0
        if n0 > 0 and n1 > 1:
            mu = (mu0 + mu1) / 2

            if self.dist == 'binom':
                return self.binomial_z_test(mu0,mu, mu1, n1)
            elif self.dist == 't':
                if n1 == 1:
                    return mu0
                else:
                    return self.t_z_test(mu0,mu, mu1, n1, sigma)
        return mu0  # Default return if conditions are not met

    def binomial_z_test(self,mu0: float, mu:float, mu1:float, n1:int) -> float:
        p = mu
        x = n1 * mu1
        # Binomial test statistic and p-value
        p_value = 2 * min(binom.cdf(x, n1, p), 1 - binom.cdf(x, n1, p))
        if p_value < self.signif_level:
            return mu1
        else:
            return mu0

    def t_z_test(self,mu0: float,mu: float, mu1: float, n1: int, sigma: float):
        s = sigma / np.sqrt(n1)  # Standard error of the mean
        # T-test statistic
        t_stat = (mu1 - mu) / s
        # Two-tailed p-value
        p_value = 2 * t.cdf(-abs(t_stat), df=n1-1)
        if p_value < self.signif_level:
            return mu1
        else:
            return mu0


#
    def calc_signif_mean(self, df: pd.DataFrame) -> pd.Series:
        num_levels = len(self.grouping_vars)
        if num_levels == 0:
            raise ValueError("Grouping variables are required to calculate significance")
        if num_levels == 1:
            # Directly return the mean from the single grouping level
            return df[f"{self.prefix}_mean_level1"]

        # Initial values for the first tztest call
        mu0 = df[f"{self.prefix}_mean_level1"].iloc[0]
        n0 = df[f"{self.prefix}_count_level1"].iloc[0]
        mu1 = df[f"{self.prefix}_mean_level2"].iloc[0]
        n1 = df[f"{self.prefix}_count_level2"].iloc[0]
        sigma = df[f"{self.prefix}_std_dev_level2"].iloc[0]

        # Run the first tztest
        mu = self.tztest(mu0, mu1, n0, n1, sigma)

        # Iterate through remaining levels
        for level in range(3, num_levels + 1):  # Start from level 3 if available
            if mu != mu0:  # If mu0 was rejected, update n0
                mu0 = mu
                n0 = n1
            # Update mu1, n1, and sigma for the next level
            mu1 = df[f"{self.prefix}_mean_level{level}"].iloc[0]
            n1 = df[f"{self.prefix}_count_level{level}"].iloc[0]
            sigma = df[f"{self.prefix}_std_dev_level{level}"].iloc[0]

            # Perform tztest for the current set
            mu = self.tztest(mu0, mu1, n0, n1, sigma)

        return pd.Series([mu], index=[f"{self.prefix}_signif_mean"])
    


    def calc_consecutive_zeros(self, df):
        # Ensure the DataFrame is sorted by id_col and time_var
        df = df.sort_values(by=[self.id_col, self.time_var])

        # Create a column that flags non-zero values
        df['non_zero'] = df[self.numeric_col] != 0

        # Group by id_col and create a cumulative sum of non-zero flags
        df['cum_non_zero'] = df.groupby(self.id_col)['non_zero'].cumsum()

        # Calculate consecutive zeros count by taking the difference of row index and last non-zero index
        df['last_non_zero_index'] = df.groupby([self.id_col, 'cum_non_zero']).cumcount()
        df['consecutive_zeros_count'] = df.groupby(self.id_col)['last_non_zero_index'].shift(1).fillna(0)

        # Clean up by removing temporary columns
        df.drop(['non_zero', 'cum_non_zero', 'last_non_zero_index'], axis=1, inplace=True)

        return df