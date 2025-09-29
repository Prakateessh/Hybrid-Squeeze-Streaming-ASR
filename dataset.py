import torch
import torch.nn as nn
import torchaudio
import json
import os

def get_vocab():
    chars = list('abcdefghijklmnopqrstuvwxyz .,?!-')
    vocab = {'<blank>': 0, '<unk>': 1}
    for i, char in enumerate(chars, start=2):
        vocab[char] = i
    return vocab

def get_idx_to_char():
    vocab = get_vocab()
    return {v: k for k, v in vocab.items()}

class ASRDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path):
        self.manifest_path = manifest_path
        self.data = []
        
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        audio_features = self.extract_features(item['audio_filepath'])
        text_tokens = self.tokenize_text(item['text'])
        
        result = {
            'audio_features': audio_features,
            'text_tokens': text_tokens,
            'text': item['text']
        }
        
        if 'prompt' in item:
            result['prompt_ids'] = torch.tensor([item.get('prompt_id', 0)], dtype=torch.long)
        
        if 'word_disfluency_labels' in item:
            result['disfluency_labels'] = torch.tensor(item['word_disfluency_labels'], dtype=torch.long)
        
        return result
    
    def extract_features(self, audio_path):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            raise RuntimeError(f"Error loading audio file {audio_path}: {e}")
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=80,
            n_fft=1024,
            hop_length=256,
            win_length=1024
        )
        
        features = mel_spectrogram(waveform).squeeze(0).transpose(0, 1)
        return features
    
    def tokenize_text(self, text):
        vocab = get_vocab()
        tokens = []
        for char in text.lower():
            if char in vocab:
                tokens.append(vocab[char])
            else:
                tokens.append(vocab['<unk>'])
        return torch.tensor(tokens, dtype=torch.long)

def collate_fn(batch):
    audio_features = [item['audio_features'] for item in batch]
    text_tokens = [item['text_tokens'] for item in batch]
    
    audio_features = torch.nn.utils.rnn.pad_sequence(audio_features, batch_first=True)
    text_tokens = torch.nn.utils.rnn.pad_sequence(text_tokens, batch_first=True)
    
    input_lengths = torch.tensor([len(x) for x in audio_features])
    target_lengths = torch.tensor([len(x) for x in text_tokens])
    
    result = {
        'audio_features': audio_features,
        'text_tokens': text_tokens,
        'targets': text_tokens,
        'input_lengths': input_lengths,
        'target_lengths': target_lengths,
        'text': [item['text'] for item in batch]
    }
    
    if 'prompt_ids' in batch[0]:
        prompt_ids = torch.stack([item['prompt_ids'] for item in batch])
        result['prompt_ids'] = prompt_ids
    
    if 'disfluency_labels' in batch[0]:
        disfluency_labels = torch.nn.utils.rnn.pad_sequence(
            [item['disfluency_labels'] for item in batch],
            batch_first=True,
            padding_value=0
        )
        result['disfluency_labels'] = disfluency_labels
    
    return result