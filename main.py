"""
Main execution script for AutoAudit Performance Visualizations
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.generate_dataset import PerformanceDatasetGenerator
from utils.data_loader import DataLoader
from visualizations.bar_charts import BarChartVisualizer
from visualizations.pie_charts import PieChartVisualizer
from visualizations.complex_matrix import ComplexMatrixVisualizer
from visualizations.line_charts import LineChartVisualizer
from visualizations.stacked_charts import StackedChartVisualizer
from models.performance_model import PerformancePredictor
from utils.config import VisualizationConfig as viz_config

def main():
    """Main execution function"""

    print("=" * 60)
    print("AutoAudit Performance Visualization System")
    print("=" * 60)

    # Step 1: Generate or load dataset
    print("\n[1/6] Generating/Loading dataset...")
    generator = PerformanceDatasetGenerator()

    # Check if dataset exists, generate if not
    data_dir = './data'
    if not os.path.exists(os.path.join(data_dir, 'performance_dataset.csv')):
        perf_data, time_data, summary = generator.generate_and_save(data_dir)
        print("Dataset generated successfully!")
    else:
        print("Dataset already exists. Loading...")

    # Step 2: Load data
    print("\n[2/6] Loading data...")
    data_loader = DataLoader(data_dir)
    perf_data, time_data, summary = data_loader.load_all_data()

    # Step 3: Train ML models
    print("\n[3/6] Training ML models...")
    predictor = PerformancePredictor()

    # Train accuracy model
    print("  - Training accuracy prediction model...")
    accuracy_results = predictor.train_accuracy_model(perf_data)
    print(f"    R² score (RF): {accuracy_results['rf']['r2']:.4f}")
    print(f"    R² score (GB): {accuracy_results['gb']['r2']:.4f}")

    # Train improvement model
    print("  - Training improvement prediction model...")
    improvement_results = predictor.train_improvement_model(perf_data)
    print(f"    R² score: {improvement_results['r2']:.4f}")
    print(f"    CV R²: {improvement_results['cv_r2_mean']:.4f} (+/- {improvement_results['cv_r2_std']:.4f})")

    # Save models
    predictor.save_models('./models/trained_model.pkl')

    # Feature importance
    print("\n  Feature Importance (Accuracy):")
    for _, row in predictor.feature_importance['accuracy'].head(5).iterrows():
        print(f"    - {row['feature']}: {row['importance']:.4f}")

    # Step 4: Generate visualizations
    print("\n[4/6] Generating visualizations...")

    # Create output directories
    os.makedirs(viz_config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(viz_config.REPORT_DIR, exist_ok=True)

    # Initialize visualizers
    bar_viz = BarChartVisualizer(data_loader)
    pie_viz = PieChartVisualizer(data_loader)
    matrix_viz = ComplexMatrixVisualizer(data_loader)
    line_viz = LineChartVisualizer(data_loader)
    stacked_viz = StackedChartVisualizer(data_loader)

    # Generate bar charts
    print("  - Generating bar charts...")
    bar_viz.plot_accuracy_comparison()
    bar_viz.plot_improvement_by_principle()
    bar_viz.plot_category_performance()
    bar_viz.plot_efficiency_metrics()

    # Generate pie charts
    print("  - Generating pie charts...")
    pie_viz.plot_system_type_distribution()
    pie_viz.plot_principle_category_distribution()
    pie_viz.plot_data_volume_distribution()

    # Generate complex matrices
    print("  - Generating complex matrices...")
    matrix_viz.plot_performance_correlation_matrix()
    matrix_viz.plot_confusion_matrix()
    matrix_viz.plot_radar_chart()
    matrix_viz.plot_improvement_heatmap()

    # Generate line charts
    print("  - Generating line charts...")
    line_viz.plot_performance_trend()
    line_viz.plot_adoption_rate_trend()
    line_viz.plot_principle_trends()
    line_viz.plot_cost_trend()

    # Generate stacked charts
    print("  - Generating stacked charts...")
    stacked_viz.plot_stacked_violations()
    stacked_viz.plot_stacked_principle_coverage()
    stacked_viz.plot_stacked_resource_allocation()

    # Step 5: Generate summary report
    print("\n[5/6] Generating summary report...")

    report_path = os.path.join(viz_config.REPORT_DIR, 'summary_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("AUTOAUDIT PERFORMANCE ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("DATASET SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Systems: {perf_data['system_id'].nunique()}\n")
        f.write(f"Total Records: {len(perf_data)}\n")
        f.write(f"Principles: {perf_data['principle'].nunique()}\n")
        f.write(f"System Types: {perf_data['system_type'].nunique()}\n\n")

        f.write("PERFORMANCE SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Baseline Avg Accuracy: {perf_data['baseline_accuracy'].mean():.4f}\n")
        f.write(f"Proposed Avg Accuracy: {perf_data['proposed_accuracy'].mean():.4f}\n")
        f.write(f"Average Improvement: {perf_data['improvement'].mean():.4f}\n\n")

        f.write("MODEL PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy Model R² (RF): {accuracy_results['rf']['r2']:.4f}\n")
        f.write(f"Accuracy Model R² (GB): {accuracy_results['gb']['r2']:.4f}\n")
        f.write(f"Improvement Model R²: {improvement_results['r2']:.4f}\n\n")

        f.write("TOP FEATURES\n")
        f.write("-" * 40 + "\n")
        for _, row in predictor.feature_importance['accuracy'].head(5).iterrows():
            f.write(f"{row['feature']}: {row['importance']:.4f}\n")

    print(f"  Report saved to {report_path}")

    # Step 6: Generate predictions for sample systems
    print("\n[6/6] Generating sample predictions...")

    # Create sample data for prediction
    sample_systems = pd.DataFrame({
        'system_type': ['Recruitment', 'Performance', 'Compensation'],
        'data_volume': ['Large', 'Medium', 'Small'],
        'principle': ['Fairness', 'Accuracy', 'Efficiency'],
        'category': ['Ethical', 'Methodical', 'Managerial'],
        'development_year': [2024, 2025, 2023]
    })

    predictions = predictor.predict_performance(sample_systems)

    if predictions:
        print("\nSample Predictions:")
        for i, (idx, row) in enumerate(sample_systems.iterrows()):
            print(f"\n  System {i+1}: {row['system_type']} - {row['principle']}")
            if 'accuracy_ensemble' in predictions:
                print(f"    Predicted Accuracy: {predictions['accuracy_ensemble'][i]:.4f}")
            if 'improvement' in predictions:
                print(f"    Predicted Improvement: {predictions['improvement'][i]:.4f}")

    print("\n" + "=" * 60)
    print("Visualization generation complete!")
    print(f"All figures saved to: {viz_config.OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()