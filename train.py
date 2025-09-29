import os
import json
import argparse
import yaml
from pathlib import Path
from hybrid_squeeze_asr import HybridSqueezeStreamingASR, RNNTLoss
from dataset import ASRDataset, collate_fn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

def train(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    os.makedirs(config['checkpoints']['save_dir'], exist_ok=True)
    
    model = HybridSqueezeStreamingASR(
        input_dim=config['model']['input_dim'],
        vocab_size=config['model']['vocab_size'],
        encoder_dim=config['model']['encoder_dim'],
        decoder_dim=config['model']['decoder_dim'],
        nhead=config['model']['nhead'],
        encoder_layers=config['model']['encoder_layers'],
        decoder_layers=config['model']['decoder_layers'],
        dropout=config['model']['dropout']
    ).to(device)
    
    criterion = RNNTLoss(blank_idx=config['loss']['blank_idx'])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['lr'],
        betas=config['optimizer']['betas'],
        eps=config['optimizer']['eps'],
        weight_decay=config['optimizer']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['scheduler']['T_max'],
        eta_min=config['scheduler']['eta_min']
    )
    
    try:
        train_dataset = ASRDataset(config['data']['train_manifest'])
        val_dataset = ASRDataset(config['data']['val_manifest'])
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run prepare_data.py first to create manifest files")
        return
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    best_val_loss = float('inf')
    no_improvement_epochs = 0
    early_stop_patience = config['training'].get('early_stop_patience', 10)
    
    for epoch in range(config['training']['num_epochs']):
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            outputs = model(
                batch['audio_features'],
                batch['text_tokens'],
                batch.get('prompt_ids', None)
            )
            
            rnnt_loss = criterion(
                outputs['joint_logits'],
                batch['targets'],
                batch['input_lengths'],
                batch['target_lengths']
            )
            
            total_loss = rnnt_loss
            
            if 'disfluency_labels' in batch and outputs.get('disfluency_logits') is not None:
                disfluency_loss = torch.nn.functional.cross_entropy(
                    outputs['disfluency_logits'].view(-1, 3),
                    batch['disfluency_labels'].view(-1)
                )
                total_loss += config['loss']['disfluency_weight'] * disfluency_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
            optimizer.step()
            
            train_loss += total_loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss.item():.4f}')
        
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = model(
                    batch['audio_features'],
                    batch['text_tokens'],
                    batch.get('prompt_ids', None)
                )
                
                rnnt_loss = criterion(
                    outputs['joint_logits'],
                    batch['targets'],
                    batch['input_lengths'],
                    batch['target_lengths']
                )
                
                val_loss += rnnt_loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement_epochs = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'config': config
            }
            torch.save(checkpoint, config['checkpoints']['best_model_path'])
            print(f'Saved best model with val loss: {avg_val_loss:.4f}')
        else:
            no_improvement_epochs += 1
        
        if no_improvement_epochs >= early_stop_patience:
            print(f'Early stopping: no improvement for {early_stop_patience} epochs')
            break

def main():
    parser = argparse.ArgumentParser(description='Train Hybrid Squeeze-Streaming ASR')
    parser.add_argument('--config', type=str, default='train_config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume training')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file {args.config} not found. Creating default config...")
        from config import create_configs
        create_configs()
    
    train(args.config)

if __name__ == "__main__":
    main()