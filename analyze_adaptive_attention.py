"""
Analyze and visualize Adaptive S²-Attn behavior
Usage: python analyze_adaptive_attention.py --model_path ./output_adaptive --context_length 16384
"""

import torch
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_attn_replace_adaptive import add_adaptive_predictor_to_model
import numpy as np

def analyze_adaptive_predictions(model, context_length=16384, num_samples=10):
    """
    Analyze what the adaptive predictors learn
    """
    results = {
        'layer_names': [],
        'group_sizes': [],
        'shift_amounts': [],
        'group_ratios': [],
        'shift_ratios': []
    }
    
    print(f"\n{'='*80}")
    print(f"Analyzing Adaptive S²-Attn Predictions (context_length={context_length})")
    print(f"{'='*80}\n")
    
    # Create dummy input
    hidden_size = model.config.hidden_size
    test_input = torch.randn(1, context_length, hidden_size).to(model.device)
    
    # Extract predictions from each layer
    for name, module in model.named_modules():
        if hasattr(module, 'adaptive_predictor') and module.adaptive_predictor is not None:
            with torch.no_grad():
                group_size, shift_amount = module.adaptive_predictor(test_input, context_length)
            
            group_ratio = group_size / context_length
            shift_ratio = shift_amount / group_size
            
            results['layer_names'].append(name)
            results['group_sizes'].append(group_size)
            results['shift_amounts'].append(shift_amount)
            results['group_ratios'].append(group_ratio)
            results['shift_ratios'].append(shift_ratio)
            
            print(f"Layer: {name:50s} | Group: {group_size:5d} ({group_ratio:.3f}) | Shift: {shift_amount:5d} ({shift_ratio:.3f})")
    
    return results

def visualize_predictions(results, save_path='adaptive_analysis.png'):
    """
    Create visualizations of adaptive predictions
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    layer_indices = range(len(results['layer_names']))
    
    # Plot 1: Group Size by Layer
    axes[0, 0].plot(layer_indices, results['group_sizes'], marker='o', linewidth=2)
    axes[0, 0].set_xlabel('Layer Index')
    axes[0, 0].set_ylabel('Group Size')
    axes[0, 0].set_title('Adaptive Group Size per Layer')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Group Size Ratio by Layer
    axes[0, 1].plot(layer_indices, results['group_ratios'], marker='s', linewidth=2, color='orange')
    axes[0, 1].axhline(y=0.25, color='r', linestyle='--', label='Original S²-Attn (0.25)')
    axes[0, 1].set_xlabel('Layer Index')
    axes[0, 1].set_ylabel('Group Size Ratio')
    axes[0, 1].set_title('Adaptive Group Size Ratio per Layer')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Shift Amount by Layer
    axes[1, 0].plot(layer_indices, results['shift_amounts'], marker='^', linewidth=2, color='green')
    axes[1, 0].set_xlabel('Layer Index')
    axes[1, 0].set_ylabel('Shift Amount')
    axes[1, 0].set_title('Adaptive Shift Amount per Layer')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Shift Ratio by Layer
    axes[1, 1].plot(layer_indices, results['shift_ratios'], marker='d', linewidth=2, color='purple')
    axes[1, 1].axhline(y=0.5, color='r', linestyle='--', label='Original S²-Attn (0.5)')
    axes[1, 1].set_xlabel('Layer Index')
    axes[1, 1].set_ylabel('Shift Ratio')
    axes[1, 1].set_title('Adaptive Shift Ratio per Layer')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to {save_path}")
    plt.close()

def compare_with_baseline(results, context_length):
    """
    Compare adaptive predictions with original S²-Attn baseline
    """
    print(f"\n{'='*80}")
    print(f"Comparison with Original S²-Attn Baseline")
    print(f"{'='*80}\n")
    
    baseline_group_size = int(context_length * 0.25)
    baseline_shift = baseline_group_size // 2
    
    avg_group_size = np.mean(results['group_sizes'])
    avg_shift = np.mean(results['shift_amounts'])
    
    print(f"Original S²-Attn:")
    print(f"  Group Size: {baseline_group_size} (ratio: 0.250)")
    print(f"  Shift Amount: {baseline_shift} (ratio: 0.500)")
    print(f"\nAdaptive S²-Attn (Average):")
    print(f"  Group Size: {avg_group_size:.1f} (ratio: {avg_group_size/context_length:.3f})")
    print(f"  Shift Amount: {avg_shift:.1f} (ratio: {avg_shift/avg_group_size:.3f})")
    print(f"\nDifferences:")
    print(f"  Group Size: {((avg_group_size - baseline_group_size) / baseline_group_size * 100):+.2f}%")
    print(f"  Shift Amount: {((avg_shift - baseline_shift) / baseline_shift * 100):+.2f}%")
    
    # Variability analysis
    group_std = np.std(results['group_sizes'])
    shift_std = np.std(results['shift_amounts'])
    
    print(f"\nVariability Across Layers:")
    print(f"  Group Size StdDev: {group_std:.1f}")
    print(f"  Shift Amount StdDev: {shift_std:.1f}")
    
    if group_std < baseline_group_size * 0.1:
        print(f"  ⚠️  Low variability - model may not be leveraging adaptivity")
    else:
        print(f"  ✅ Good variability - model is adapting per layer")

def analyze_content_sensitivity(model, tokenizer, texts, context_length=8192):
    """
    Test how predictions change with different content types
    """
    print(f"\n{'='*80}")
    print(f"Content Sensitivity Analysis")
    print(f"{'='*80}\n")
    
    results = {}
    
    for content_type, text in texts.items():
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", max_length=context_length, truncation=True, padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Get hidden states from first layer
        with torch.no_grad():
            outputs = model.model.layers[0](
                model.model.embed_tokens(inputs['input_ids']),
                attention_mask=inputs['attention_mask']
            )
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Get prediction from first layer's adaptive predictor
        first_layer = model.model.layers[0].self_attn
        if hasattr(first_layer, 'adaptive_predictor'):
            with torch.no_grad():
                group_size, shift_amount = first_layer.adaptive_predictor(hidden_states, hidden_states.shape[1])
            
            results[content_type] = {
                'group_size': group_size,
                'shift_amount': shift_amount,
                'group_ratio': group_size / hidden_states.shape[1],
                'shift_ratio': shift_amount / group_size
            }
            
            print(f"{content_type:20s} | Group: {group_size:5d} ({results[content_type]['group_ratio']:.3f}) | "
                  f"Shift: {shift_amount:5d} ({results[content_type]['shift_ratio']:.3f})")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Analyze Adaptive S²-Attn')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--context_length', type=int, default=16384, help='Context length to analyze')
    parser.add_argument('--output_prefix', type=str, default='adaptive_analysis', help='Output file prefix')
    parser.add_argument('--test_content', action='store_true', help='Test content sensitivity')
    args = parser.parse_args()
    
    print(f"\n🔍 Loading model from {args.model_path}...")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    
    # Load tokenizer if testing content
    tokenizer = None
    if args.test_content:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Check if model has adaptive predictors
    has_adaptive = any(hasattr(m, 'adaptive_predictor') for _, m in model.named_modules())
    
    if not has_adaptive:
        print("⚠️  Model doesn't have adaptive predictors. Loading them now...")
        model = add_adaptive_predictor_to_model(model)
    
    # Analyze predictions
    results = analyze_adaptive_predictions(model, args.context_length)
    
    # Create visualizations
    visualize_predictions(results, f'{args.output_prefix}.png')
    
    # Compare with baseline
    compare_with_baseline(results, args.context_length)
    
    # Test content sensitivity if requested
    if args.test_content and tokenizer:
        test_texts = {
            'Code': 'def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n' * 100,
            'Prose': 'It was the best of times, it was the worst of times. ' * 200,
            'Technical': 'The Transformer architecture consists of encoder and decoder stacks. ' * 200,
            'Repetitive': 'Hello world. ' * 500,
        }
        content_results = analyze_content_sensitivity(model, tokenizer, test_texts, min(args.context_length, 8192))
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Results saved to {args.output_prefix}.png")

if __name__ == "__main__":
    main()