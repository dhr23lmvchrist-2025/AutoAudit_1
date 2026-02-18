"""
Dataset Generator for AutoAudit Performance Visualization
Generates synthetic but realistic performance data based on the comparison metrics
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os


class PerformanceDatasetGenerator:
    """
    Generates synthetic dataset simulating AutoAudit vs Baseline performance
    across multiple systems and principles
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)

        # Configuration based on the comparison data
        self.n_systems = 500  # Total systems to simulate
        self.principles = [
            'Veracity', 'Accuracy', 'Validity', 'Relevancy', 'Quality',
            'Efficiency', 'Fairness', 'Transparency', 'Accountability'
        ]

        self.categories = ['Methodical', 'Methodical', 'Methodical',
                           'Managerial', 'Managerial', 'Managerial',
                           'Ethical', 'Ethical', 'Ethical']

        # Baseline accuracies from the paper
        self.baseline_accuracies = {
            'Veracity': 0.892, 'Accuracy': 0.915, 'Validity': 0.798,
            'Relevancy': 0.763, 'Quality': 0.827, 'Efficiency': 0.841,
            'Fairness': 0.884, 'Transparency': 0.816, 'Accountability': 0.742
        }

        # AutoAudit accuracies from the paper
        self.proposed_accuracies = {
            'Veracity': 0.967, 'Accuracy': 0.982, 'Validity': 0.884,
            'Relevancy': 0.851, 'Quality': 0.913, 'Efficiency': 0.936,
            'Fairness': 0.952, 'Transparency': 0.908, 'Accountability': 0.865
        }

        # Standard deviations for realistic variation
        self.baseline_std = 0.042
        self.proposed_std = 0.018

    def generate_performance_data(self):
        """Generate synthetic performance data for multiple systems"""

        data = []

        for system_id in range(self.n_systems):
            # Generate system metadata
            system_type = np.random.choice(['Recruitment', 'Performance', 'Compensation',
                                            'Training', 'Retention'], p=[0.3, 0.25, 0.2, 0.15, 0.1])
            data_volume = np.random.choice(['Small', 'Medium', 'Large', 'Very Large'],
                                           p=[0.2, 0.4, 0.3, 0.1])
            development_year = np.random.randint(2018, 2026)

            # Generate per-principle performance
            for i, principle in enumerate(self.principles):
                category = self.categories[i]

                # Generate baseline score with realistic variation
                baseline_mean = self.baseline_accuracies[principle]
                baseline_score = np.random.normal(baseline_mean, self.baseline_std)
                baseline_score = np.clip(baseline_score, 0.6, 0.98)

                # Generate proposed score with realistic variation
                proposed_mean = self.proposed_accuracies[principle]
                proposed_score = np.random.normal(proposed_mean, self.proposed_std)
                proposed_score = np.clip(proposed_score, 0.75, 0.995)

                # Calculate improvement
                improvement = proposed_score - baseline_score

                # Generate additional metrics
                precision_baseline = baseline_score * np.random.uniform(0.95, 1.05)
                precision_proposed = proposed_score * np.random.uniform(0.98, 1.02)

                recall_baseline = baseline_score * np.random.uniform(0.94, 1.06)
                recall_proposed = proposed_score * np.random.uniform(0.97, 1.03)

                f1_baseline = 2 * (precision_baseline * recall_baseline) / (
                            precision_baseline + recall_baseline + 1e-10)
                f1_proposed = 2 * (precision_proposed * recall_proposed) / (
                            precision_proposed + recall_proposed + 1e-10)

                # Response time (hours)
                response_time_baseline = np.random.gamma(shape=2, scale=20)  # 40 hours avg
                response_time_proposed = np.random.gamma(shape=1.5, scale=1.7)  # 2.5 hours avg

                # Error rate
                error_rate_baseline = 1 - baseline_score
                error_rate_proposed = 1 - proposed_score

                # Scalability metrics
                scalability_data = np.random.choice([1000, 10000, 100000, 1000000, 10000000],
                                                    p=[0.1, 0.2, 0.3, 0.25, 0.15])

                # Cost
                cost_baseline = np.random.uniform(4000, 8000)
                cost_proposed = np.random.uniform(50, 150)

                # Create record
                record = {
                    'system_id': f'SYS_{system_id:04d}',
                    'system_type': system_type,
                    'data_volume': data_volume,
                    'development_year': development_year,
                    'principle': principle,
                    'category': category,
                    'baseline_accuracy': baseline_score,
                    'proposed_accuracy': proposed_score,
                    'improvement': improvement,
                    'precision_baseline': precision_baseline,
                    'precision_proposed': precision_proposed,
                    'recall_baseline': recall_baseline,
                    'recall_proposed': recall_proposed,
                    'f1_baseline': f1_baseline,
                    'f1_proposed': f1_proposed,
                    'response_time_baseline': response_time_baseline,
                    'response_time_proposed': response_time_proposed,
                    'error_rate_baseline': error_rate_baseline,
                    'error_rate_proposed': error_rate_proposed,
                    'scalability_data_size': scalability_data,
                    'cost_baseline': cost_baseline,
                    'cost_proposed': cost_proposed
                }
                data.append(record)

        return pd.DataFrame(data)

    def generate_time_series_data(self, n_months=24):
        """Generate time series data for trend analysis"""

        dates = pd.date_range(start='2024-01-01', periods=n_months, freq='M')
        time_data = []

        for i, date in enumerate(dates):
            # Simulate improving performance over time
            time_factor = 1 + (i / n_months) * 0.1  # 10% improvement over period

            for principle in self.principles:
                record = {
                    'date': date,
                    'principle': principle,
                    'baseline_trend': self.baseline_accuracies[principle] * np.random.normal(1, 0.02) * time_factor,
                    'proposed_trend': self.proposed_accuracies[principle] * np.random.normal(1, 0.01) * (
                                time_factor ** 1.2),
                    'adoption_rate': np.random.uniform(0.1, 0.9) * (i / n_months) ** 0.8,
                    'detection_rate': np.random.uniform(0.7, 0.98) * time_factor
                }
                time_data.append(record)

        return pd.DataFrame(time_data)

    def generate_and_save(self, output_dir='./data'):
        """Generate datasets and save to CSV"""

        os.makedirs(output_dir, exist_ok=True)

        # Generate main performance dataset
        print("Generating performance dataset...")
        perf_data = self.generate_performance_data()
        perf_path = os.path.join(output_dir, 'performance_dataset.csv')
        perf_data.to_csv(perf_path, index=False)
        print(f"Saved to {perf_path}")

        # Generate time series dataset
        print("Generating time series dataset...")
        time_data = self.generate_time_series_data()
        time_path = os.path.join(output_dir, 'time_series_dataset.csv')
        time_data.to_csv(time_path, index=False)
        print(f"Saved to {time_path}")

        # Generate summary statistics
        summary = perf_data.groupby('principle').agg({
            'baseline_accuracy': 'mean',
            'proposed_accuracy': 'mean',
            'improvement': 'mean',
            'precision_baseline': 'mean',
            'precision_proposed': 'mean',
            'recall_baseline': 'mean',
            'recall_proposed': 'mean',
            'f1_baseline': 'mean',
            'f1_proposed': 'mean'
        }).round(4)

        summary_path = os.path.join(output_dir, 'summary_statistics.csv')
        summary.to_csv(summary_path)
        print(f"Summary statistics saved to {summary_path}")

        return perf_data, time_data, summary


if __name__ == "__main__":
    generator = PerformanceDatasetGenerator()
    perf_data, time_data, summary = generator.generate_and_save()
    print("\nDataset generation complete!")
    print(f"Performance dataset shape: {perf_data.shape}")
    print(f"Time series dataset shape: {time_data.shape}")
    print("\nSummary Statistics:")
    print(summary)