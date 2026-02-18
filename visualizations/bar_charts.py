"""
Bar Chart Visualizations for AutoAudit Performance Comparison
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from utils.config import VisualizationConfig as viz_config


class BarChartVisualizer:
    """Generate bar chart visualizations for performance comparison"""

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.performance_data = data_loader.performance_data
        viz_config.set_style()

        # Create output directory
        os.makedirs(viz_config.OUTPUT_DIR, exist_ok=True)

    def plot_accuracy_comparison(self):
        """Plot accuracy comparison bar chart (Figure 1 from paper)"""

        # Get performance by principle
        perf_by_principle = self.data_loader.get_performance_by_principle('accuracy')

        if perf_by_principle is None:
            print("No data available")
            return

        principles = perf_by_principle['principle'].values
        x = np.arange(len(principles))
        width = 0.35

        fig, ax = plt.subplots(figsize=viz_config.LARGE_FIGURE_SIZE)

        # Create bars
        bars1 = ax.bar(x - width / 2, perf_by_principle['baseline_accuracy'],
                       width, label='Baseline (Manual Audit)',
                       color=viz_config.BASELINE_COLOR, alpha=0.8)
        bars2 = ax.bar(x + width / 2, perf_by_principle['proposed_accuracy'],
                       width, label='AutoAudit (Proposed)',
                       color=viz_config.PROPOSED_COLOR, alpha=0.8)

        # Customize chart
        ax.set_xlabel('Principles', fontsize=viz_config.AXIS_FONT_SIZE)
        ax.set_ylabel('Accuracy Score', fontsize=viz_config.AXIS_FONT_SIZE)
        ax.set_title('Accuracy Comparison: Baseline vs AutoAudit by Principle',
                     fontsize=viz_config.TITLE_FONT_SIZE, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(principles, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.set_ylim(0.7, 1.0)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # Save figure
        output_path = os.path.join(viz_config.OUTPUT_DIR, 'bar_accuracy_comparison.png')
        plt.savefig(output_path, dpi=viz_config.FIGURE_DPI, bbox_inches='tight')
        plt.show()
        print(f"Saved to {output_path}")

        return fig

    def plot_improvement_by_principle(self):
        """Plot improvement bars showing delta between baseline and proposed"""

        perf_by_principle = self.data_loader.get_performance_by_principle('accuracy')

        if perf_by_principle is None:
            return

        # Sort by improvement
        perf_sorted = perf_by_principle.sort_values('improvement', ascending=True)

        fig, ax = plt.subplots(figsize=viz_config.FIGURE_SIZE)

        # Create horizontal bar chart
        colors = [viz_config.PROPOSED_COLOR if x > 0 else viz_config.BASELINE_COLOR
                  for x in perf_sorted['improvement']]

        bars = ax.barh(perf_sorted['principle'], perf_sorted['improvement'],
                       color=colors, alpha=0.8)

        # Customize
        ax.set_xlabel('Improvement (Accuracy Gain)', fontsize=viz_config.AXIS_FONT_SIZE)
        ax.set_title('Performance Improvement by Principle',
                     fontsize=viz_config.TITLE_FONT_SIZE, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f'{width:.3f}',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3 if width > 0 else -30, 0),
                        textcoords="offset points",
                        ha='left' if width > 0 else 'right',
                        va='center', fontsize=9)

        plt.tight_layout()

        output_path = os.path.join(viz_config.OUTPUT_DIR, 'bar_improvement_by_principle.png')
        plt.savefig(output_path, dpi=viz_config.FIGURE_DPI, bbox_inches='tight')
        plt.show()

        return fig

    def plot_category_performance(self):
        """Plot performance by category (Methodical, Managerial, Ethical)"""

        if self.performance_data is None:
            return

        # Group by category
        category_perf = self.performance_data.groupby('category').agg({
            'baseline_accuracy': 'mean',
            'proposed_accuracy': 'mean'
        }).reset_index()

        fig, ax = plt.subplots(figsize=viz_config.SMALL_FIGURE_SIZE)

        x = np.arange(len(category_perf))
        width = 0.35

        bars1 = ax.bar(x - width / 2, category_perf['baseline_accuracy'],
                       width, label='Baseline', color=viz_config.BASELINE_COLOR, alpha=0.8)
        bars2 = ax.bar(x + width / 2, category_perf['proposed_accuracy'],
                       width, label='AutoAudit', color=viz_config.PROPOSED_COLOR, alpha=0.8)

        ax.set_xlabel('Principle Category', fontsize=viz_config.AXIS_FONT_SIZE)
        ax.set_ylabel('Average Accuracy', fontsize=viz_config.AXIS_FONT_SIZE)
        ax.set_title('Performance by Principle Category',
                     fontsize=viz_config.TITLE_FONT_SIZE, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(category_perf['category'])
        ax.legend()
        ax.set_ylim(0.7, 1.0)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        output_path = os.path.join(viz_config.OUTPUT_DIR, 'bar_category_performance.png')
        plt.savefig(output_path, dpi=viz_config.FIGURE_DPI, bbox_inches='tight')
        plt.show()

        return fig

    def plot_efficiency_metrics(self):
        """Plot efficiency metrics comparison (time, cost)"""

        if self.performance_data is None:
            return

        # Calculate mean metrics
        metrics = {
            'Response Time (hours)': {
                'baseline': self.performance_data['response_time_baseline'].mean(),
                'proposed': self.performance_data['response_time_proposed'].mean()
            },
            'Error Rate (%)': {
                'baseline': self.performance_data['error_rate_baseline'].mean() * 100,
                'proposed': self.performance_data['error_rate_proposed'].mean() * 100
            },
            'Cost ($)': {
                'baseline': self.performance_data['cost_baseline'].mean(),
                'proposed': self.performance_data['cost_proposed'].mean()
            }
        }

        fig, axes = plt.subplots(1, 3, figsize=viz_config.LARGE_FIGURE_SIZE)

        for i, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[i]

            x = [0, 1]
            heights = [values['baseline'], values['proposed']]
            colors = [viz_config.BASELINE_COLOR, viz_config.PROPOSED_COLOR]

            bars = ax.bar(x, heights, color=colors, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(['Baseline', 'AutoAudit'])
            ax.set_title(metric_name, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        plt.suptitle('Efficiency Metrics Comparison', fontsize=viz_config.TITLE_FONT_SIZE, fontweight='bold')
        plt.tight_layout()

        output_path = os.path.join(viz_config.OUTPUT_DIR, 'bar_efficiency_metrics.png')
        plt.savefig(output_path, dpi=viz_config.FIGURE_DPI, bbox_inches='tight')
        plt.show()

        return fig