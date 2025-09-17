"""Experiment runner for comparing different model configurations."""

import os
import torch
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional, Tuple
import json

from ..config.config import Config, ExperimentConfig
from ..data.tokenizer import Tokenizer
from ..data.data_loader import POSDataset, create_data_loaders
from ..models.embeddings import Embeddings
from ..models.encoders import EncoderFFN
from ..models.codebook import GumbelCodebook
from ..models.decoders import DecoderFFN, DecoderBiLSTM, CharDecoder
from ..models.tagging_model import TaggingModel
from ..training.trainer import Trainer
from ..training.evaluator import Evaluator
from ..utils.data_utils import read_conllu_file, prepare_data_indices
from ..utils.visualization import (
    plot_training_curves, plot_confusion_matrix, 
    create_top_words_table, plot_codebook_usage
)
from ..utils.metrics import calculate_m1_accuracy


class ExperimentRunner:
    """Class to run and manage experiments."""
    
    def __init__(self, experiment_config: ExperimentConfig):
        """
        Initialize experiment runner.
        
        Args:
            experiment_config: Configuration for experiments
        """
        self.exp_config = experiment_config
        self.device = torch.device("cuda" if torch.cuda.is_available() and experiment_config.use_cuda else "cpu")
        print(f"Using device: {self.device}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(experiment_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(experiment_config.seed)

    def load_data(self) -> Tuple[List, List, List, Dict, Dict, Dict, Dict, Dict, Dict]:
        """
        Load and prepare datasets.
        
        Returns:
            Tuple containing datasets and index mappings
        """
        # Load CoNLL-U files
        train_file = os.path.join(self.exp_config.data_dir, self.exp_config.train_file)
        dev_file = os.path.join(self.exp_config.data_dir, self.exp_config.dev_file)
        test_file = os.path.join(self.exp_config.data_dir, self.exp_config.test_file)
        
        train_sents, train_labels = read_conllu_file(train_file)
        dev_sents, _ = read_conllu_file(dev_file)
        test_sents, _ = read_conllu_file(test_file)
        
        # Prepare index mappings
        l2i, i2l, v2i, i2v, c2i, i2c = prepare_data_indices(train_sents, train_labels)
        
        print(f"Loaded data:")
        print(f"  Train sentences: {len(train_sents)}")
        print(f"  Dev sentences: {len(dev_sents)}")
        print(f"  Test sentences: {len(test_sents)}")
        print(f"  Vocabulary size: {len(v2i)}")
        print(f"  Label set size: {len(l2i)}")
        print(f"  Character set size: {len(c2i)}")
        
        return (train_sents, dev_sents, test_sents, 
                l2i, i2l, v2i, i2v, c2i, i2c)

    def create_model_components(self, config: Config, v2i: Dict, c2i: Dict, l2i: Dict) -> Tuple:
        """
        Create model components based on configuration.
        
        Returns:
            Tuple of model components
        """
        # BERT tokenizer and model
        bert_tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name)
        bert_model = AutoModel.from_pretrained(config.bert_model_name)
        
        # Embeddings
        embeddings = Embeddings(
            num_chars=len(c2i),
            config=config,
            bert_model=bert_model,
            num_layers=1
        )
        
        # Encoder
        encoder = EncoderFFN(
            emb_hidden=config.emb_hidden,
            num_tag=config.num_tag
        )
        
        # Codebook
        codebook = GumbelCodebook(
            num_tag=config.num_tag,
            tag_dim=config.tag_dim
        )
        
        # Decoders
        vocab_decoder_ffn = DecoderFFN(
            tag_dim=config.tag_dim,
            vocab_size=len(v2i)
        )
        
        vocab_decoder_bilstm = DecoderBiLSTM(
            tag_dim=config.tag_dim,
            dec_hidden=config.dec_hidden,
            vocab_size=len(v2i),
            num_layers=config.dec_layers
        )
        
        char_decoder = None
        if config.use_char_architecture:
            char_decoder = CharDecoder(
                tag_dim=config.tag_dim,
                hidden_size=config.dec_hidden,
                char_vocab_size=len(c2i),
                max_char_len=config.max_word_len
            )
        
        return (bert_tokenizer, embeddings, encoder, codebook, 
                vocab_decoder_ffn, vocab_decoder_bilstm, char_decoder)

    def run_experiment(self, 
                      config: Config,
                      experiment_name: str,
                      decoder_type: str = "bilstm",
                      use_char_decoder: bool = False,
                      datasets: Optional[Tuple] = None,
                      mappings: Optional[Tuple] = None) -> Dict:
        """
        Run a single experiment.
        
        Args:
            config: Model configuration
            experiment_name: Name of the experiment
            decoder_type: Type of decoder ("ffn" or "bilstm")
            use_char_decoder: Whether to use character decoder
            datasets: Pre-loaded datasets (optional)
            mappings: Pre-loaded index mappings (optional)
            
        Returns:
            Dictionary with experiment results
        """
        print(f"\n{'='*50}")
        print(f"Running Experiment: {experiment_name}")
        print(f"{'='*50}")
        
        # Load data if not provided
        if datasets is None or mappings is None:
            (train_sents, dev_sents, test_sents, 
             l2i, i2l, v2i, i2v, c2i, i2c) = self.load_data()
        else:
            train_sents, dev_sents, test_sents = datasets
            l2i, i2l, v2i, i2v, c2i, i2c = mappings
        
        # Create tokenizer
        bert_tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name)
        tokenizer = Tokenizer(v2i, l2i, c2i, config, bert_tokenizer)
        
        # Create datasets
        train_dataset = POSDataset(train_sents, tokenizer, config)
        dev_dataset = POSDataset(dev_sents, tokenizer, config)
        test_dataset = POSDataset(test_sents, tokenizer, config)
        
        # Create data loaders
        train_loader, dev_loader, test_loader = create_data_loaders(
            train_dataset, dev_dataset, test_dataset, config.batch_size
        )
        
        # Create model components
        (bert_tokenizer, embeddings, encoder, codebook, 
         vocab_decoder_ffn, vocab_decoder_bilstm, char_decoder) = self.create_model_components(
            config, v2i, c2i, l2i
        )
        
        # Select decoder
        if decoder_type == "ffn":
            vocab_decoder = vocab_decoder_ffn
        else:
            vocab_decoder = vocab_decoder_bilstm
        
        if not use_char_decoder:
            char_decoder = None
        
        # Create model
        model = TaggingModel(
            config=config,
            embeddings=embeddings,
            encoder=encoder,
            codebook=codebook,
            vocab_decoder=vocab_decoder,
            char_decoder=char_decoder,
            vocab_size=len(v2i)
        ).to(self.device)
        
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create trainer and train
        trainer = Trainer(model, config, self.device, c2i)
        save_dir = os.path.join(self.exp_config.output_dir, experiment_name)
        
        training_results = trainer.train(
            train_loader, dev_loader, save_dir, experiment_name
        )
        
        # Evaluate on test set
        evaluator = Evaluator(model, config, self.device, c2i)
        test_results = evaluator.evaluate(test_loader, l2i, i2l, mode='test')
        
        # Create visualizations
        self._create_visualizations(
            experiment_name, training_results, test_results, 
            save_dir, i2l, i2v, c2i
        )
        
        # Analyze codebook usage
        codebook_stats = evaluator.analyze_codebook_usage(test_loader)
        plot_codebook_usage(codebook_stats, experiment_name, save_dir)
        
        # Compile results
        results = {
            'experiment_name': experiment_name,
            'config': config.__dict__,
            'training_losses': training_results[0],
            'validation_losses': training_results[1],
            'diversity_losses': training_results[2],
            'vocab_losses': training_results[3],
            'char_losses': training_results[4],
            'test_loss': test_results['avg_loss'],
            'm1_accuracy': test_results.get('m1_accuracy', None),
            'codebook_stats': codebook_stats,
            'save_dir': save_dir
        }
        
        # Save results
        self._save_results(results, save_dir)
        
        print(f"Experiment {experiment_name} completed!")
        if 'm1_accuracy' in test_results:
            print(f"M1 Accuracy: {test_results['m1_accuracy']:.2f}%")
        
        return results

    def _create_visualizations(self, 
                             experiment_name: str,
                             training_results: Tuple,
                             test_results: Dict,
                             save_dir: str,
                             i2l: Dict,
                             i2v: Dict,
                             c2i: Dict) -> None:
        """Create and save visualizations."""
        # Training curves
        plot_training_curves(
            experiment_name=experiment_name,
            loss_values=training_results[0],
            val_values=training_results[1],
            vocab_loss_values=training_results[3],
            div_loss_values=training_results[2],
            char_loss_values=training_results[4],
            save_dir=save_dir
        )
        
        # Confusion matrices (if supervised data available)
        if test_results.get('m1_dict') is not None:
            plot_confusion_matrix(
                test_results['m1_dict'], experiment_name, i2l, 
                is_m1=True, save_dir=save_dir
            )
        
        if test_results.get('count_dict') is not None:
            plot_confusion_matrix(
                test_results['count_dict'], experiment_name, i2l, 
                is_m1=False, save_dir=save_dir
            )
        
        # Top words table
        if test_results.get('word_tag_counts') is not None:
            create_top_words_table(
                test_results['word_tag_counts'],
                i2v,
                i2l,
                test_results.get('pred_to_m1'),
                experiment_name,
                save_dir
            )

    def _save_results(self, results: Dict, save_dir: str) -> None:
        """Save experiment results to JSON file."""
        # Convert non-serializable objects
        serializable_results = {}
        for key, value in results.items():
            if key == 'config':
                serializable_results[key] = value
            elif isinstance(value, (list, dict, str, int, float, bool)) or value is None:
                serializable_results[key] = value
            else:
                serializable_results[key] = str(value)
        
        results_file = os.path.join(save_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

    def run_baseline_experiments(self) -> Dict[str, Dict]:
        """
        Run baseline experiments with different configurations.
        
        Returns:
            Dictionary mapping experiment names to results
        """
        # Load data once
        data_and_mappings = self.load_data()
        datasets = data_and_mappings[:3]
        mappings = data_and_mappings[3:]
        
        experiments = {}
        
        # Experiment 1: Baseline (BiLSTM decoder, no char components)
        config1 = Config()
        config1.use_char_architecture = False
        config1.epochs = 10  # Reduced for testing
        experiments['baseline'] = self.run_experiment(
            config1, 'baseline', 'bilstm', False, datasets, mappings
        )
        
        # Experiment 2: FFN decoder
        config2 = Config()
        config2.use_char_architecture = False
        config2.epochs = 10
        experiments['ffn_decoder'] = self.run_experiment(
            config2, 'ffn_decoder', 'ffn', False, datasets, mappings
        )
        
        # Experiment 3: Character embeddings
        config3 = Config()
        config3.use_char_architecture = True
        config3.epochs = 10
        experiments['char_embeddings'] = self.run_experiment(
            config3, 'char_embeddings', 'ffn', False, datasets, mappings
        )
        
        # Experiment 4: Character decoder
        config4 = Config()
        config4.use_char_architecture = True
        config4.vocab_loss_weight = 0.5
        config4.epochs = 10
        experiments['char_decoder'] = self.run_experiment(
            config4, 'char_decoder', 'ffn', True, datasets, mappings
        )
        
        return experiments

    def run_hyperparameter_search(self, base_config: Config, param_grid: Dict) -> Dict[str, Dict]:
        """
        Run hyperparameter search experiments.
        
        Args:
            base_config: Base configuration
            param_grid: Dictionary with parameter ranges
            
        Returns:
            Dictionary mapping experiment names to results
        """
        data_and_mappings = self.load_data()
        datasets = data_and_mappings[:3]
        mappings = data_and_mappings[3:]
        
        experiments = {}
        
        # Generate parameter combinations
        import itertools
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for i, param_combination in enumerate(itertools.product(*param_values)):
            # Create config for this combination
            config = Config()
            config.__dict__.update(base_config.__dict__)
            
            exp_name_parts = []
            for param_name, param_value in zip(param_names, param_combination):
                setattr(config, param_name, param_value)
                exp_name_parts.append(f"{param_name}_{param_value}")
            
            exp_name = f"hyperparam_{i}_{'_'.join(exp_name_parts)}"
            
            try:
                experiments[exp_name] = self.run_experiment(
                    config, exp_name, 'bilstm', False, datasets, mappings
                )
            except Exception as e:
                print(f"Experiment {exp_name} failed: {e}")
                continue
        
        return experiments

    def compare_experiments(self, experiments: Dict[str, Dict]) -> Dict:
        """
        Compare results across experiments.
        
        Args:
            experiments: Dictionary mapping experiment names to results
            
        Returns:
            Comparison summary
        """
        if not experiments:
            return {}
        
        comparison = {
            'experiment_names': list(experiments.keys()),
            'metrics': {}
        }
        
        # Extract comparable metrics
        for exp_name, results in experiments.items():
            comparison['metrics'][exp_name] = {
                'final_train_loss': results['training_losses'][-1] if results['training_losses'] else None,
                'final_val_loss': results['validation_losses'][-1] if results['validation_losses'] else None,
                'test_loss': results.get('test_loss'),
                'm1_accuracy': results.get('m1_accuracy'),
                'active_codes': results['codebook_stats'].get('num_active_tags'),
                'entropy': results['codebook_stats'].get('entropy')
            }
        
        # Find best performing experiment
        best_exp = None
        best_m1_acc = -1
        
        for exp_name, metrics in comparison['metrics'].items():
            if metrics['m1_accuracy'] is not None and metrics['m1_accuracy'] > best_m1_acc:
                best_m1_acc = metrics['m1_accuracy']
                best_exp = exp_name
        
        comparison['best_experiment'] = best_exp
        comparison['best_m1_accuracy'] = best_m1_acc
        
        return comparison

    def save_experiment_summary(self, experiments: Dict[str, Dict], comparison: Dict, filename: str = None) -> None:
        """Save experiment summary to file."""
        if filename is None:
            filename = os.path.join(self.exp_config.output_dir, 'experiment_summary.json')
        
        summary = {
            'experiments': {name: {
                'config': exp['config'],
                'final_results': {
                    'test_loss': exp.get('test_loss'),
                    'm1_accuracy': exp.get('m1_accuracy'),
                    'codebook_active_codes': exp['codebook_stats'].get('num_active_tags'),
                    'codebook_entropy': exp['codebook_stats'].get('entropy')
                }
            } for name, exp in experiments.items()},
            'comparison': comparison
        }
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Experiment summary saved to {filename}")

    def create_experiment_report(self, experiments: Dict[str, Dict]) -> str:
        """
        Create a formatted experiment report.
        
        Args:
            experiments: Dictionary mapping experiment names to results
            
        Returns:
            Formatted report string
        """
        report_lines = [
            "# Experiment Report",
            "=" * 50,
            "",
            f"Total experiments run: {len(experiments)}",
            f"Device used: {self.device}",
            ""
        ]
        
        # Summary table
        report_lines.extend([
            "## Summary Results",
            "",
            "| Experiment | Test Loss | M1 Accuracy | Active Codes | Entropy |",
            "|------------|-----------|-------------|--------------|---------|"
        ])
        
        for exp_name, results in experiments.items():
            test_loss = f"{results.get('test_loss', 'N/A'):.4f}" if results.get('test_loss') else "N/A"
            m1_acc = f"{results.get('m1_accuracy', 'N/A'):.2f}%" if results.get('m1_accuracy') else "N/A"
            active_codes = results['codebook_stats'].get('num_active_tags', 'N/A')
            entropy = f"{results['codebook_stats'].get('entropy', 'N/A'):.3f}" if results['codebook_stats'].get('entropy') else "N/A"
            
            report_lines.append(f"| {exp_name} | {test_loss} | {m1_acc} | {active_codes} | {entropy} |")
        
        report_lines.extend([
            "",
            "## Individual Experiment Details",
            ""
        ])
        
        # Detailed results for each experiment
        for exp_name, results in experiments.items():
            report_lines.extend([
                f"### {exp_name}",
                "",
                "**Configuration:**",
            ])
            
            for key, value in results['config'].items():
                report_lines.append(f"- {key}: {value}")
            
            report_lines.extend([
                "",
                "**Results:**",
                f"- Final training loss: {results['training_losses'][-1]:.4f}" if results['training_losses'] else "- Training loss: N/A",
                f"- Final validation loss: {results['validation_losses'][-1]:.4f}" if results['validation_losses'] else "- Validation loss: N/A",
                f"- Test loss: {results.get('test_loss', 'N/A')}",
                f"- M1 accuracy: {results.get('m1_accuracy', 'N/A')}%" if results.get('m1_accuracy') else "- M1 accuracy: N/A",
                "",
                "**Codebook Statistics:**"
            ])
            
            for key, value in results['codebook_stats'].items():
                report_lines.append(f"- {key}: {value}")
            
            report_lines.extend(["", "---", ""])
        
        return "\n".join(report_lines)