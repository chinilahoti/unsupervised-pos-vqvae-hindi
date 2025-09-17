#!/usr/bin/env python3
"""Simple test runner to verify the codebase works."""

import sys
import traceback


def run_test_module(module_name, test_function=None):
    """Run a specific test module."""
    try:
        print(f"\n{'='*50}")
        print(f"Running {module_name}")
        print(f"{'='*50}")
        
        if module_name == "config":
            from tests.test_config import (
                test_config_creation, test_config_modification, 
                test_experiment_config, test_get_device
            )
            test_config_creation()
            test_config_modification()
            test_experiment_config()
            test_get_device()
            
        elif module_name == "data_utils":
            from tests.test_data_utils import (
                test_read_conllu_file, test_item_indexer, test_prepare_data_indices
            )
            test_read_conllu_file()
            test_item_indexer()
            test_prepare_data_indices()
            
        elif module_name == "models":
            from tests.test_models import (
                test_encoder_ffn, test_gumbel_codebook, test_decoder_ffn,
                test_decoder_bilstm, test_model_integration, test_gradients_flow
            )
            test_encoder_ffn()
            test_gumbel_codebook()
            test_decoder_ffn()
            test_decoder_bilstm()
            test_model_integration()
            test_gradients_flow()
            
        elif module_name == "integration":
            from tests.test_integration import test_full_pipeline_minimal
            test_full_pipeline_minimal()
            
        print(f"✓ {module_name} tests PASSED")
        
    except Exception as e:
        print(f"✗ {module_name} tests FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests."""
    print("Starting test suite for POS Tagger")
    print("This will test individual components with small data")
    
    test_modules = [
        "config",
        "data_utils", 
        "models",
        "integration"  # This one downloads a small model, may take time
    ]
    
    passed = 0
    failed = 0
    
    for module in test_modules:
        if run_test_module(module):
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("🎉 All tests passed! Your code structure is working correctly.")
        print("\nNext steps:")
        print("1. Test with your actual data files")
        print("2. Run a small experiment: python main.py --experiment baseline --epochs 1")
        print("3. Check the output directory for results")
    else:
        print("❌ Some tests failed. Please fix the issues before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
