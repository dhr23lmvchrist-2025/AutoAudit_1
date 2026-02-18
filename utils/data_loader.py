"""
Data Loader Utility for AutoAudit Visualizations
"""

import pandas as pd
import numpy as np
import os


class DataLoader:
    """Load and preprocess performance data for visualizations"""

    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        self.performance_data = None
        self.time_series_data = None
        self.summary_stats = None

    def load_all_data(self):
        """Load all datasets"""
        self.performance_data = self.load_performance_data()
        self.time_series_data = self.load_time_series_data()
        self.summary_stats = self.load_summary_stats()
        return self.performance_data, self.time_series_data, self.summary_stats

    def load_performance_data(self):
        """Load main performance dataset"""
        path = os.path.join(self.data_dir, 'performance_dataset.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded performance data: {df.shape[0]} records")
            return df
        else:
            print(f"Warning: {path} not found. Generating sample data...")
            from data.generate_dataset import PerformanceDatasetGenerator
            generator = PerformanceDatasetGenerator()
            df, _, _ = generator.generate_and_save()
            return df

    def load_time_series_data(self):
        """Load time series dataset"""
        path = os.path.join(self.data_dir, 'time_series_dataset.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['date'] = pd.to_datetime(df['date'])
            print(f"Loaded time series data: {df.shape[0]} records")
            return df
        else:
            print(f"Warning: {path} not found")
            return None

    def load_summary_stats(self):
        """Load summary statistics"""
        path = os.path.join(self.data_dir, 'summary_statistics.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df
        else:
            return None

    def get_principles_by_category(self):
        """Group principles by category"""
        return {
            'Methodical': ['Veracity', 'Accuracy', 'Validity'],
            'Managerial': ['Relevancy', 'Quality', 'Efficiency'],
            'Ethical': ['Fairness', 'Transparency', 'Accountability']
        }

    def get_system_type_distribution(self):
        """Get distribution of systems by type"""
        if self.performance_data is not None:
            return self.performance_data['system_type'].value_counts()
        return None

    def get_performance_by_principle(self, metric='accuracy'):
        """Get mean performance by principle"""
        if self.performance_data is not None:
            if metric == 'accuracy':
                baseline_col = 'baseline_accuracy'
                proposed_col = 'proposed_accuracy'
            elif metric == 'precision':
                baseline_col = 'precision_baseline'
                proposed_col = 'precision_proposed'
            elif metric == 'recall':
                baseline_col = 'recall_baseline'
                proposed_col = 'recall_proposed'
            elif metric == 'f1':
                baseline_col = 'f1_baseline'
                proposed_col = 'f1_proposed'
            else:
                return None

            result = self.performance_data.groupby('principle').agg({
                baseline_col: 'mean',
                proposed_col: 'mean'
            }).reset_index()

            result['improvement'] = result[proposed_col] - result[baseline_col]
            return result

        return None