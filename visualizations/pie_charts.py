"""
Pie Chart Visualizations for AutoAudit Performance Data
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from utils.config import VisualizationConfig as viz_config


class PieChartVisualizer:
    """Generate pie chart visualizations"""

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.performance_data = data_loader.performance_data
        viz_config.set_style()

        os.makedirs(viz_config.OUTPUT_DIR, exist_ok=True)

    def plot_system_type_distribution(self):
        """Plot distribution of system types in the dataset"""

        system_dist = self.data_loader.get_system_type_distribution()

        if system_dist is None:
            return

        fig, ax = plt.subplots(figsize=viz_config.SMALL_FIGURE_SIZE)

        # Create pie chart
        wedges, texts, autotexts = ax.pie(system_dist.values,
                                          labels=system_dist.index,
                                          autopct='%1.1f%%',
                                          startangle=90,
                                          colors=plt.cm.Set3(np.linspace(0, 1, len(system_dist))))

        # Customize
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title('Distribution of Systems by HR Function',
                     fontsize=viz_config.TITLE_FONT_SIZE, fontweight='bold')

        plt.tight_layout()

        output_path = os.path.join(viz_config.OUTPUT_DIR, 'pie_system_distribution.png')
        plt.savefig(output_path, dpi=viz_config.FIGURE_DPI, bbox_inches='tight')
        plt.show()

        return fig

    def plot_principle_category_distribution(self):
        """Plot distribution of principles by category"""

        if self.performance_data is None:
            return

        category_counts = self.performance_data.groupby('category')['principle'].nunique()

        fig, ax = plt.subplots(figsize=viz_config.SMALL_FIGURE_SIZE)

        colors = [viz_config.METHODICAL_COLOR, viz_config.MANAGERIAL_COLOR, viz_config.ETHICAL_COLOR]

        wedges, texts, autotexts = ax.pie(category_counts.values,
                                          labels=category_counts.index,
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90,
                                          explode=(0.05, 0.05, 0.05))

        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title('Distribution of Principles by Category',
                     fontsize=viz_config.TITLE_FONT_SIZE, fontweight='bold')

        plt.tight_layout()

        output_path = os.path.join(viz_config.OUTPUT_DIR, 'pie_category_distribution.png')
        plt.savefig(output_path, dpi=viz_config.FIGURE_DPI, bbox_inches='tight')
        plt.show()

        return fig

    def plot_data_volume_distribution(self):
        """Plot distribution of data volumes"""

        if self.performance_data is None:
            return

        volume_counts = self.performance_data['data_volume'].value_counts()

        fig, ax = plt.subplots(figsize=viz_config.SMALL_FIGURE_SIZE)

        wedges, texts, autotexts = ax.pie(volume_counts.values,
                                          labels=volume_counts.index,
                                          autopct='%1.1f%%',
                                          startangle=90,
                                          colors=plt.cm.Paired(np.linspace(0, 1, len(volume_counts))))

        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_color('white')

        ax.set_title('Distribution by Data Volume',
                     fontsize=viz_config.TITLE_FONT_SIZE, fontweight='bold')

        plt.tight_layout()

        output_path = os.path.join(viz_config.OUTPUT_DIR, 'pie_data_volume.png')
        plt.savefig(output_path, dpi=viz_config.FIGURE_DPI, bbox_inches='tight')
        plt.show()

        return fig