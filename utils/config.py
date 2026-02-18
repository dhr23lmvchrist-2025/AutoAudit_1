"""
Configuration settings for AutoAudit Visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns


class VisualizationConfig:
    """Configuration class for visualization settings"""

    # Color schemes
    BASELINE_COLOR = '#FF6B6B'  # Coral red
    PROPOSED_COLOR = '#4ECDC4'  # Turquoise
    METHODICAL_COLOR = '#FFB347'  # Orange
    MANAGERIAL_COLOR = '#5D9B9B'  # Teal
    ETHICAL_COLOR = '#9B59B6'  # Purple

    PRINCIPLE_COLORS = {
        'Veracity': '#FF6B6B',
        'Accuracy': '#4ECDC4',
        'Validity': '#45B7D1',
        'Relevancy': '#96CEB4',
        'Quality': '#FFEEAD',
        'Efficiency': '#D4A5A5',
        'Fairness': '#9B59B6',
        'Transparency': '#3498DB',
        'Accountability': '#E67E22'
    }

    CATEGORY_COLORS = {
        'Methodical': '#FFB347',
        'Managerial': '#5D9B9B',
        'Ethical': '#9B59B6'
    }

    # Figure settings
    FIGURE_SIZE = (12, 8)
    SMALL_FIGURE_SIZE = (10, 6)
    LARGE_FIGURE_SIZE = (16, 10)

    # Font settings
    TITLE_FONT_SIZE = 16
    AXIS_FONT_SIZE = 12
    LEGEND_FONT_SIZE = 10
    ANNOTATION_FONT_SIZE = 9

    # Style settings
    @staticmethod
    def set_style():
        """Set global plotting style"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        plt.rcParams['figure.figsize'] = VisualizationConfig.FIGURE_SIZE
        plt.rcParams['font.size'] = VisualizationConfig.AXIS_FONT_SIZE
        plt.rcParams['axes.titlesize'] = VisualizationConfig.TITLE_FONT_SIZE
        plt.rcParams['legend.fontsize'] = VisualizationConfig.LEGEND_FONT_SIZE

    # Output settings
    OUTPUT_DIR = './outputs/figures'
    REPORT_DIR = './outputs/reports'
    FIGURE_FORMAT = 'png'
    FIGURE_DPI = 300