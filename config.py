import yaml

train_config = {
    'model': {
        'input_dim': 80,
        'vocab_size': 1000,
        'encoder_dim': 512,
        'decoder_dim': 512,
        'nhead': 8,
        'encoder_layers': 12,
        'decoder_layers': 6,
        'dropout': 0.1
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'num_epochs': 100,
        'warmup_steps': 5000,
        'max_grad_norm': 1.0,
        'early_stop_patience': 10
    },
    'data': {
        'train_manifest': 'data/train-clean-100_manifest.json',
        'val_manifest': 'data/dev-clean_manifest.json',
        'test_manifest': 'data/test-clean_manifest.json',
        'sample_rate': 16000,
        'n_mels': 80,
        'n_fft': 1024,
        'hop_length': 256,
        'win_length': 1024
    },
    'optimizer': {
        'name': 'AdamW',
        'lr': 1e-4,
        'betas': [0.9, 0.999],
        'eps': 1e-8,
        'weight_decay': 0.01
    },
    'scheduler': {
        'name': 'CosineAnnealingLR',
        'T_max': 100,
        'eta_min': 1e-6
    },
    'loss': {
        'blank_idx': 0,
        'disfluency_weight': 0.1
    },
    'checkpoints': {
        'save_dir': 'checkpoints',
        'best_model_path': 'checkpoints/best_model.pt'
    }
}

eval_config = {
    'model': {
        'checkpoint_path': 'checkpoints/best_model.pt',
        'input_dim': 80,
        'vocab_size': 1000,
        'encoder_dim': 512,
        'decoder_dim': 512
    },
    'evaluation': {
        'test_manifest': 'data/test-clean_manifest.json',
        'batch_size': 16,
        'beam_width': 5,
        'max_length': 512
    },
    'metrics': {
        'compute_wer': True,
        'compute_cer': True,
        'compute_bleu': True,
        'disfluency_detection': True
    }
}

streaming_config = {
    'streaming': {
        'chunk_size': 160,
        'hop_length': 80,
        'lookahead_frames': 80,
        'cache_size': 1000
    },
    'model': {
        'checkpoint_path': 'checkpoints/streaming_model.pt',
        'quantization': 'int8',
        'optimization': 'cuda_graphs'
    },
    'deployment': {
        'max_batch_size': 8,
        'max_sequence_length': 2048,
        'rtf_target': 0.25
    }
}

def create_configs():
    import os
    os.makedirs('configs', exist_ok=True)
    
    with open('configs/train_config.yaml', 'w') as f:
        yaml.dump(train_config, f, default_flow_style=False)
    
    with open('configs/eval_config.yaml', 'w') as f:
        yaml.dump(eval_config, f, default_flow_style=False)
    
    with open('configs/streaming_config.yaml', 'w') as f:
        yaml.dump(streaming_config, f, default_flow_style=False)
    
    with open('train_config.yaml', 'w') as f:
        yaml.dump(train_config, f, default_flow_style=False)
    
    with open('eval_config.yaml', 'w') as f:
        yaml.dump(eval_config, f, default_flow_style=False)
    
    with open('streaming_config.yaml', 'w') as f:
        yaml.dump(streaming_config, f, default_flow_style=False)
    
    print("Configuration files created successfully!")
    print("- train_config.yaml")
    print("- eval_config.yaml") 
    print("- streaming_config.yaml")
    print("- configs/train_config.yaml")
    print("- configs/eval_config.yaml")
    print("- configs/streaming_config.yaml")

if __name__ == "__main__":
    create_configs()