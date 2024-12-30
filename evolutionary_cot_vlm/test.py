from models import load_model
from evals import evaluate_model
import argparse

'''
# Evaluate on validation set (default)
python test.py --model blip2 --benchmark chartqa

# Evaluate on training set
python test.py --model blip2 --benchmark chartqa --split train

# Evaluate on test set
python test.py --model blip2 --benchmark chartqa --split test
'''

def main():
    parser = argparse.ArgumentParser(description='Evaluate a multimodal model on benchmarks')
    parser.add_argument('--model', type=str, required=True, 
                      choices=['blip2', 'llava', 'minigpt4', 'claude', 'molmo'],
                      help='Name of the model to evaluate')
    parser.add_argument('--benchmark', type=str, required=True,
                      choices=['chartqa', 'vqav2', 'mmmu'],
                      help='Benchmark dataset to evaluate on')
    parser.add_argument('--split', type=str, default='validation',
                      choices=['train', 'validation', 'test'],
                      help='Dataset split to evaluate on (default: validation)')
    parser.add_argument('--num_samples', type=int, default=None,
                      help='Number of samples to evaluate (default: all)')
    parser.add_argument('--prefix', type=str, default="",
                      help='Prefix to add before each question')
    
    args = parser.parse_args()

    # Load model and processor
    print(f"Loading {args.model}...")
    model, processor = load_model(args.model)

    # Run evaluation
    print(f"Evaluating on {args.benchmark} ({args.split} split)...")
    metrics = evaluate_model(
        model=model,
        processor=processor,
        benchmark=args.benchmark,
        split=args.split,
        num_samples=args.num_samples,
        prefix=args.prefix
    )

    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Correct samples: {metrics['correct_samples']}/{metrics['total_samples']}")

if __name__ == "__main__":
    main()