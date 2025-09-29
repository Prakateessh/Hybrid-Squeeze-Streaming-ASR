import argparse
import json
import yaml
import torch
from pathlib import Path
from hybrid_squeeze_asr import HybridSqueezeStreamingASR, inference
from dataset import ASRDataset, collate_fn, get_idx_to_char
from torch.utils.data import DataLoader

def evaluate_model(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = HybridSqueezeStreamingASR(
        input_dim=config['model']['input_dim'],
        vocab_size=config['model']['vocab_size'],
        encoder_dim=config['model']['encoder_dim'],
        decoder_dim=config['model']['decoder_dim']
    ).to(device)
    
    try:
        checkpoint = torch.load(config['model']['checkpoint_path'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {config['model']['checkpoint_path']}")
    except FileNotFoundError:
        print(f"Checkpoint not found: {config['model']['checkpoint_path']}")
        print("Please train the model first using train.py")
        return
    
    model.eval()
    
    try:
        test_dataset = ASRDataset(config['evaluation']['test_manifest'])
    except FileNotFoundError:
        print(f"Test manifest not found: {config['evaluation']['test_manifest']}")
        print("Please run prepare_data.py first to create manifest files")
        return
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    idx_to_char = get_idx_to_char()
    
    predictions = []
    references = []
    disfluency_predictions = []
    disfluency_references = []
    
    print("Starting evaluation...")
    
    for batch_idx, batch in enumerate(test_loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        with torch.no_grad():
            decoded_tokens, disfluency_logits = inference(
                model, 
                batch['audio_features'],
                batch.get('prompt_ids', None),
                config['evaluation']['max_length']
            )
            
            for i in range(len(batch['text'])):
                pred_text = decode_tokens(decoded_tokens[i], idx_to_char)
                ref_text = batch['text'][i]
                
                predictions.append(pred_text)
                references.append(ref_text)
                
                if 'disfluency_labels' in batch and disfluency_logits is not None:
                    disfluency_pred = torch.argmax(disfluency_logits[i], dim=-1)
                    disfluency_predictions.append(disfluency_pred.cpu().numpy())
                    disfluency_references.append(batch['disfluency_labels'][i].cpu().numpy())
        
        if batch_idx % 10 == 0:
            print(f"Processed {batch_idx * config['evaluation']['batch_size']} samples")
    
    metrics = compute_metrics(predictions, references, disfluency_predictions, disfluency_references)
    
    print("\nEvaluation Results:")
    print(f"WER: {metrics['wer']:.4f}")
    print(f"CER: {metrics['cer']:.4f}")
    if metrics['disfluency_f1'] is not None:
        print(f"Disfluency F1: {metrics['disfluency_f1']:.4f}")
    
    save_results(predictions, references, metrics, 'evaluation_results.json')

def decode_tokens(tokens, idx_to_char):
    text = ''
    for token in tokens:
        if token.item() in idx_to_char and token.item() != 0:
            text += idx_to_char[token.item()]
    return text.strip()

def compute_wer(predictions, references):
    total_errors = 0
    total_words = 0
    
    for pred, ref in zip(predictions, references):
        pred_words = pred.split()
        ref_words = ref.split()
        
        total_words += len(ref_words)
        
        if len(ref_words) == 0:
            continue
        
        d = [[0] * (len(pred_words) + 1) for _ in range(len(ref_words) + 1)]
        
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(pred_words) + 1):
            d[0][j] = j
        
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(pred_words) + 1):
                if ref_words[i-1] == pred_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
        
        total_errors += d[len(ref_words)][len(pred_words)]
    
    return total_errors / total_words if total_words > 0 else 0

def compute_cer(predictions, references):
    total_errors = 0
    total_chars = 0
    
    for pred, ref in zip(predictions, references):
        total_chars += len(ref)
        
        if len(ref) == 0:
            continue
        
        d = [[0] * (len(pred) + 1) for _ in range(len(ref) + 1)]
        
        for i in range(len(ref) + 1):
            d[i][0] = i
        for j in range(len(pred) + 1):
            d[0][j] = j
        
        for i in range(1, len(ref) + 1):
            for j in range(1, len(pred) + 1):
                if ref[i-1] == pred[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
        
        total_errors += d[len(ref)][len(pred)]
    
    return total_errors / total_chars if total_chars > 0 else 0

def compute_disfluency_f1(predictions, references):
    if not predictions or not references:
        return None
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred, ref in zip(predictions, references):
        min_len = min(len(pred), len(ref))
        for i in range(min_len):
            if pred[i] == 1 and ref[i] == 1:
                true_positives += 1
            elif pred[i] == 1 and ref[i] == 0:
                false_positives += 1
            elif pred[i] == 0 and ref[i] == 1:
                false_negatives += 1
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1

def compute_metrics(predictions, references, disfluency_predictions=None, disfluency_references=None):
    wer = compute_wer(predictions, references)
    cer = compute_cer(predictions, references)
    
    disfluency_f1 = None
    if disfluency_predictions and disfluency_references:
        disfluency_f1 = compute_disfluency_f1(disfluency_predictions, disfluency_references)
    
    return {
        'wer': wer,
        'cer': cer,
        'disfluency_f1': disfluency_f1
    }

def save_results(predictions, references, metrics, output_file):
    results = {
        'metrics': metrics,
        'examples': [
            {'prediction': pred, 'reference': ref}
            for pred, ref in zip(predictions[:10], references[:10])
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")

def benchmark_inference_speed(model, test_loader, device):
    import time
    
    model.eval()
    total_time = 0
    total_samples = 0
    
    print("Benchmarking inference speed...")
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            start_time = time.time()
            
            outputs = model(batch['audio_features'])
            
            end_time = time.time()
            
            total_time += (end_time - start_time)
            total_samples += batch['audio_features'].size(0)
            
            if total_samples >= 100:
                break
    
    avg_inference_time = total_time / total_samples
    rtf = avg_inference_time / (0.1)  # Assuming average 0.1s per frame
    
    print(f"Average inference time per sample: {avg_inference_time:.4f}s")
    print(f"Real-time factor (RTF): {rtf:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Hybrid Squeeze-Streaming ASR')
    parser.add_argument('--config', type=str, default='eval_config.yaml', help='Path to evaluation config')
    parser.add_argument('--benchmark', action='store_true', help='Run inference speed benchmark')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file {args.config} not found. Creating default config...")
        from config import create_configs
        create_configs()
    
    evaluate_model(args.config)
    
    if args.benchmark:
        print("\nRunning inference speed benchmark...")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = HybridSqueezeStreamingASR(
            input_dim=config['model']['input_dim'],
            vocab_size=config['model']['vocab_size'],
            encoder_dim=config['model']['encoder_dim'],
            decoder_dim=config['model']['decoder_dim']
        ).to(device)
        
        try:
            checkpoint = torch.load(config['model']['checkpoint_path'], map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            test_dataset = ASRDataset(config['evaluation']['test_manifest'])
            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=collate_fn
            )
            
            benchmark_inference_speed(model, test_loader, device)
        except (FileNotFoundError, KeyError) as e:
            print(f"Error during benchmarking: {e}")

if __name__ == "__main__":
    import os
    main()