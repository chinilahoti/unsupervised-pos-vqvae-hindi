"""Main entry point for POS tagging experiments."""

import argparse
import os

from pos_inductor.config import Config, ExperimentConfig
from pos_inductor.experiments import ExperimentRunner


def main():
    """Main function to run POS tagging experiments."""
    parser = argparse.ArgumentParser(description='Run POS Tagging Experiments')

    # Experiment arguments
    parser.add_argument('--experiment', type=str, default='baseline',
                        choices=['baseline', 'all', 'hyperparameter_search'],
                        help='Type of experiment to run')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing CoNLL-U files')
    parser.add_argument('--output_dir', type=str, default='experiments',
                        help='Directory to save experiment results')

    # Model arguments
    parser.add_argument('--bert_model', type=str, default='google/muril-base-cased',
                        help='BERT models to use')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=6e-5,
                        help='Learning rate')

    # Architecture arguments
    parser.add_argument('--num_tags', type=int, default=100,
                        help='Number of discrete tags')
    parser.add_argument('--tag_dim', type=int, default=50,
                        help='Dimension of tag representations')
    parser.add_argument('--diversity_weight', type=float, default=0.6,
                        help='Weight for diversity loss')

    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Create experiment configuration
    exp_config = ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_cuda=(args.device == 'cuda') if args.device != 'auto' else True,
        seed=args.seed
    )

    # Create base models configuration
    base_config = Config()
    base_config.bert_model_name = args.bert_model
    base_config.epochs = args.epochs
    base_config.batch_size = args.batch_size
    base_config.learning_rate = args.learning_rate
    base_config.num_tag = args.num_tags
    base_config.tag_dim = args.tag_dim
    base_config.diversity_weight = args.diversity_weight

    # Initialize experiment runner
    runner = ExperimentRunner(exp_config)

    # Run experiments based on type
    if args.experiment == 'baseline':
        print("Running baseline experiment...")
        data_and_mappings = runner.load_data()
        datasets = data_and_mappings[:3]
        mappings = data_and_mappings[3:]

        results = runner.run_experiment(
            base_config, 'baseline', 'bilstm', False, datasets, mappings
        )

        print(f"\nBaseline experiment completed!")
        if results.get('m1_accuracy'):
            print(f"M1 Accuracy: {results['m1_accuracy']:.2f}%")

    elif args.experiment == 'all':
        print("Running all baseline experiments...")
        experiments = runner.run_baseline_experiments()

        # Compare experiments
        comparison = runner.compare_experiments(experiments)

        # Save summary
        runner.save_experiment_summary(experiments, comparison)

        # Create report
        report = runner.create_experiment_report(experiments)
        report_file = os.path.join(exp_config.output_dir, 'experiment_report.md')
        with open(report_file, 'w') as f:
            f.write(report)

        print(f"\nAll experiments completed!")
        print(f"Best experiment: {comparison.get('best_experiment')}")
        print(f"Best M1 accuracy: {comparison.get('best_m1_accuracy', 'N/A')}")
        print(f"Report saved to: {report_file}")

    elif args.experiment == 'hyperparameter_search':
        print("Running hyperparameter search...")

        # Define parameter grid
        param_grid = {
            'learning_rate': [1e-5, 5e-5, 1e-4],
            'diversity_weight': [0.1, 0.6, 1.0],
            'gumbel_temperature': [0.5, 1.0, 2.0]
        }

        experiments = runner.run_hyperparameter_search(base_config, param_grid)

        # Compare results
        comparison = runner.compare_experiments(experiments)
        runner.save_experiment_summary(experiments, comparison,
                                       filename=os.path.join(exp_config.output_dir, 'hyperparameter_results.json'))

        print(f"\nHyperparameter search completed!")
        print(f"Best configuration: {comparison.get('best_experiment')}")

    else:
        raise ValueError(f"Unknown experiment type: {args.experiment}")


if __name__ == "__main__":
    main()
