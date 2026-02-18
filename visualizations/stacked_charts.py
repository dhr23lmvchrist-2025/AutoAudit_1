"""
Stacked Chart Visualizations for AutoAudit Performance Data
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from utils.config import VisualizationConfig as viz_config


class StackedChartVisualizer:
    """Generate stacked chart visualizations"""

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.performance_data = data_loader.performance_data
        viz_config.set_style()

        os.makedirs(viz_config.OUTPUT_DIR, exist_ok=True)

    def plot_stacked_violations(self):
        """Plot stacked bar chart of violation types"""

        # Simulate violation data
        violation_types = ['Data Quality', 'Bias', 'Accuracy', 'Transparency',
                           'Accountability', 'Efficiency', 'Relevancy']

        # Violations detected by each method
        baseline_detected = [45, 32, 28, 18, 12, 22, 15]
        proposed_detected = [52, 48, 35, 42, 38, 41, 33]
        total_violations = [58, 51, 38, 45, 40, 44, 35]

        # Calculate undetected
        baseline_missed = [total - base for total, base in zip(total_violations, baseline_detected)]
        proposed_missed = [total - prop for total, prop in zip(total_violations, proposed_detected)]

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Baseline stacked chart
        ax1 = axes[0]
        ax1.bar(violation_types, baseline_detected, label='Detected',
                color=viz_config.PROPOSED_COLOR, alpha=0.8)
        ax1.bar(violation_types, baseline_missed, bottom=baseline_detected,
                label='Missed', color=viz_config.BASELINE_COLOR, alpha=0.6)

        ax1.set_xlabel('Violation Type', fontsize=viz_config.AXIS_FONT_SIZE)
        ax1.set_ylabel('Number of Violations', fontsize=viz_config.AXIS_FONT_SIZE)
        ax1.set_title('Baseline: Violation Detection', fontweight='bold')
        ax1.legend()
        ax1.set_xticklabels(violation_types, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')

        # Add percentage labels
        for i, (detected, missed) in enumerate(zip(baseline_detected, baseline_missed)):
            total = detected + missed
            pct = (detected / total) * 100
            ax1.text(i, detected + missed / 2, f'{pct:.1f}%',
                     ha='center', va='center', fontsize=9, fontweight='bold')

        # Proposed stacked chart
        ax2 = axes[1]
        ax2.bar(violation_types, proposed_detected, label='Detected',
                color=viz_config.PROPOSED_COLOR, alpha=0.8)
        ax2.bar(violation_types, proposed_missed, bottom=proposed_detected,
                label='Missed', color=viz_config.BASELINE_COLOR, alpha=0.6)

        ax2.set_xlabel('Violation Type', fontsize=viz_config.AXIS_FONT_SIZE)
        ax2.set_ylabel('Number of Violations', fontsize=viz_config.AXIS_FONT_SIZE)
        ax2.set_title('AutoAudit: Violation Detection', fontweight='bold')
        ax2.legend()
        ax2.set_xticklabels(violation_types, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add percentage labels
        for i, (detected, missed) in enumerate(zip(proposed_detected, proposed_missed)):
            total = detected + missed
            pct = (detected / total) * 100
            ax2.text(i, detected + missed / 2, f'{pct:.1f}%',
                     ha='center', va='center', fontsize=9, fontweight='bold')

        plt.suptitle('Violation Detection Performance: Stacked Comparison',
                     fontsize=viz_config.TITLE_FONT_SIZE, fontweight='bold')
        plt.tight_layout()

        output_path = os.path.join(viz_config.OUTPUT_DIR, 'stacked_violations.png')
        plt.savefig(output_path, dpi=viz_config.FIGURE_DPI, bbox_inches='tight')
        plt.show()

        return fig

    def plot_stacked_principle_coverage(self):
        """Plot stacked bar chart of principle coverage across systems"""

        if self.performance_data is None:
            return

        # Calculate coverage (systems addressing each principle)
        coverage = self.performance_data.groupby('principle').size().reset_index(name='count')
        coverage = coverage.sort_values('count', ascending=False)

        # Simulate full coverage data (systems that fully comply)
        # This is synthetic data for demonstration
        np.random.seed(42)
        full_compliance = []
        partial_compliance = []

        for principle in coverage['principle']:
            total = coverage[coverage['principle'] == principle]['count'].values[0]
            full = int(total * np.random.uniform(0.6, 0.9))
            partial = total - full
            full_compliance.append(full)
            partial_compliance.append(partial)

        fig, ax = plt.subplots(figsize=viz_config.LARGE_FIGURE_SIZE)

        # Create stacked bars
        x = np.arange(len(coverage['principle']))

        ax.bar(x, full_compliance, label='Full Compliance',
               color=viz_config.PROPOSED_COLOR, alpha=0.8)
        ax.bar(x, partial_compliance, bottom=full_compliance,
               label='Partial/No Compliance', color=viz_config.BASELINE_COLOR, alpha=0.6)

        ax.set_xlabel('Principles', fontsize=viz_config.AXIS_FONT_SIZE)
        ax.set_ylabel('Number of Systems', fontsize=viz_config.AXIS_FONT_SIZE)
        ax.set_title('Principle Coverage Across Systems',
                     fontsize=viz_config.TITLE_FONT_SIZE, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(coverage['principle'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add total labels
        for i, (full, partial) in enumerate(zip(full_compliance, partial_compliance)):
            total = full + partial
            pct = (full / total) * 100
            ax.text(i, total + 5, f'{pct:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()

        output_path = os.path.join(viz_config.OUTPUT_DIR, 'stacked_coverage.png')
        plt.savefig(output_path, dpi=viz_config.FIGURE_DPI, bbox_inches='tight')
        plt.show()

        return fig

    def plot_stacked_resource_allocation(self):
        """Plot stacked area chart of resource allocation over time"""

        months = np.arange(1, 25)

        # Simulate resource allocation (hours) for different activities
        baseline_data_prep = 15 * np.ones(len(months))
        baseline_model_eval = 12 * np.ones(len(months))
        baseline_ethics_review = 8 * np.ones(len(months))
        baseline_reporting = 5 * np.ones(len(months))

        # AutoAudit: automated tasks reduce manual effort
        proposed_data_prep = 5 * np.exp(-0.05 * months) + 2
        proposed_model_eval = 3 * np.exp(-0.08 * months) + 1
        proposed_ethics_review = 2 * np.exp(-0.1 * months) + 0.5
        proposed_reporting = 1 * np.ones(len(months))

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Baseline stacked area
        ax1 = axes[0]
        ax1.stackplot(months,
                      [baseline_data_prep, baseline_model_eval,
                       baseline_ethics_review, baseline_reporting],
                      labels=['Data Preparation', 'Model Evaluation',
                              'Ethics Review', 'Reporting'],
                      alpha=0.7,
                      colors=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])

        ax1.set_xlabel('Months', fontsize=viz_config.AXIS_FONT_SIZE)
        ax1.set_ylabel('Hours per Assessment', fontsize=viz_config.AXIS_FONT_SIZE)
        ax1.set_title('Baseline: Manual Resource Allocation', fontweight='bold')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_ylim(0, 50)
        ax1.grid(True, alpha=0.3)

        # Proposed stacked area
        ax2 = axes[1]
        ax2.stackplot(months,
                      [proposed_data_prep, proposed_model_eval,
                       proposed_ethics_review, proposed_reporting],
                      labels=['Data Preparation', 'Model Evaluation',
                              'Ethics Review', 'Reporting'],
                      alpha=0.7,
                      colors=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])

        ax2.set_xlabel('Months', fontsize=viz_config.AXIS_FONT_SIZE)
        ax2.set_ylabel('Hours per Assessment', fontsize=viz_config.AXIS_FONT_SIZE)
        ax2.set_title('AutoAudit: Automated Resource Allocation', fontweight='bold')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.set_ylim(0, 15)
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Resource Allocation Over Time: Manual vs Automated',
                     fontsize=viz_config.TITLE_FONT_SIZE, fontweight='bold')
        plt.tight_layout()

        output_path = os.path.join(viz_config.OUTPUT_DIR, 'stacked_resource.png')
        plt.savefig(output_path, dpi=viz_config.FIGURE_DPI, bbox_inches='tight')
        plt.show()

        return fig