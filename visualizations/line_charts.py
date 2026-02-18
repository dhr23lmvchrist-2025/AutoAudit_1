"""
Line Chart Visualizations for AutoAudit Performance Trends
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from utils.config import VisualizationConfig as viz_config


class LineChartVisualizer:
    """Generate line chart visualizations for temporal trends"""

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.performance_data = data_loader.performance_data
        self.time_series_data = data_loader.time_series_data
        viz_config.set_style()

        os.makedirs(viz_config.OUTPUT_DIR, exist_ok=True)

    def plot_performance_trend(self):
        """Plot performance trends over time"""

        if self.time_series_data is None:
            print("No time series data available")
            return

        fig, ax = plt.subplots(figsize=viz_config.LARGE_FIGURE_SIZE)

        # Group by date and calculate mean
        trend_data = self.time_series_data.groupby('date').agg({
            'baseline_trend': 'mean',
            'proposed_trend': 'mean'
        }).reset_index()

        ax.plot(trend_data['date'], trend_data['baseline_trend'],
                marker='o', linewidth=2, label='Baseline',
                color=viz_config.BASELINE_COLOR)
        ax.plot(trend_data['date'], trend_data['proposed_trend'],
                marker='s', linewidth=2, label='AutoAudit',
                color=viz_config.PROPOSED_COLOR)

        ax.set_xlabel('Date', fontsize=viz_config.AXIS_FONT_SIZE)
        ax.set_ylabel('Performance Score', fontsize=viz_config.AXIS_FONT_SIZE)
        ax.set_title('Performance Trend Over Time: Baseline vs AutoAudit',
                     fontsize=viz_config.TITLE_FONT_SIZE, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.75, 1.0)

        # Format x-axis
        plt.xticks(rotation=45)

        plt.tight_layout()

        output_path = os.path.join(viz_config.OUTPUT_DIR, 'line_performance_trend.png')
        plt.savefig(output_path, dpi=viz_config.FIGURE_DPI, bbox_inches='tight')
        plt.show()

        return fig

    def plot_adoption_rate_trend(self):
        """Plot adoption rate trend over time"""

        if self.time_series_data is None:
            return

        fig, ax = plt.subplots(figsize=viz_config.FIGURE_SIZE)

        # Group by date
        adoption_data = self.time_series_data.groupby('date')['adoption_rate'].mean().reset_index()

        ax.fill_between(adoption_data['date'], 0, adoption_data['adoption_rate'],
                        alpha=0.3, color=viz_config.PROPOSED_COLOR)
        ax.plot(adoption_data['date'], adoption_data['adoption_rate'],
                linewidth=2, color=viz_config.PROPOSED_COLOR, marker='o')

        ax.set_xlabel('Date', fontsize=viz_config.AXIS_FONT_SIZE)
        ax.set_ylabel('Adoption Rate', fontsize=viz_config.AXIS_FONT_SIZE)
        ax.set_title('AutoAudit Adoption Rate Trend',
                     fontsize=viz_config.TITLE_FONT_SIZE, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.xticks(rotation=45)
        plt.tight_layout()

        output_path = os.path.join(viz_config.OUTPUT_DIR, 'line_adoption_trend.png')
        plt.savefig(output_path, dpi=viz_config.FIGURE_DPI, bbox_inches='tight')
        plt.show()

        return fig

    def plot_principle_trends(self):
        """Plot trends for each principle category"""

        if self.time_series_data is None:
            return

        principles_by_cat = self.data_loader.get_principles_by_category()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, (category, principles) in enumerate(principles_by_cat.items()):
            ax = axes[idx]

            for principle in principles:
                principle_data = self.time_series_data[
                    self.time_series_data['principle'] == principle
                    ].sort_values('date')

                if not principle_data.empty:
                    ax.plot(principle_data['date'], principle_data['proposed_trend'],
                            linewidth=2, label=principle,
                            color=viz_config.PRINCIPLE_COLORS.get(principle, None))

            ax.set_xlabel('Date', fontsize=viz_config.AXIS_FONT_SIZE)
            ax.set_ylabel('Performance Score', fontsize=viz_config.AXIS_FONT_SIZE)
            ax.set_title(f'{category} Principles Performance Trend', fontweight='bold')
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.75, 1.0)

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.suptitle('Performance Trends by Principle Category',
                     fontsize=viz_config.TITLE_FONT_SIZE, fontweight='bold')
        plt.tight_layout()

        output_path = os.path.join(viz_config.OUTPUT_DIR, 'line_principle_trends.png')
        plt.savefig(output_path, dpi=viz_config.FIGURE_DPI, bbox_inches='tight')
        plt.show()

        return fig

    def plot_cost_trend(self):
        """Plot cost trend over time"""

        # Simulate cost reduction over time
        months = np.arange(1, 25)

        # Learning curve effect: costs decrease over time
        baseline_cost = 6000 * np.exp(-0.03 * months) + 2000 * np.random.randn(len(months)) * 0.1
        proposed_cost = 75 * np.exp(-0.1 * months) + 50 * np.random.randn(len(months)) * 0.1

        fig, ax = plt.subplots(figsize=viz_config.FIGURE_SIZE)

        ax.plot(months, baseline_cost, linewidth=2, label='Baseline',
                color=viz_config.BASELINE_COLOR)
        ax.plot(months, proposed_cost, linewidth=2, label='AutoAudit',
                color=viz_config.PROPOSED_COLOR)

        ax.set_xlabel('Months', fontsize=viz_config.AXIS_FONT_SIZE)
        ax.set_ylabel('Cost per Assessment ($)', fontsize=viz_config.AXIS_FONT_SIZE)
        ax.set_title('Cost Trend Over Time: Learning Curve Effect',
                     fontsize=viz_config.TITLE_FONT_SIZE, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Use log scale for better visualization
        ax.set_yscale('log')

        plt.tight_layout()

        output_path = os.path.join(viz_config.OUTPUT_DIR, 'line_cost_trend.png')
        plt.savefig(output_path, dpi=viz_config.FIGURE_DPI, bbox_inches='tight')
        plt.show()

        return fig