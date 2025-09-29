import argparse
import torch
import torchaudio
import yaml
import os
from pathlib import Path
from hybrid_squeeze_asr import HybridSqueezeStreamingASR, inference, StreamingProcessor
from dataset import get_vocab, get_idx_to_char

def transcribe_audio(audio_path, config_path, prompt=None):
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
    except FileNotFoundError:
        print(f"Checkpoint not found: {config['model']['checkpoint_path']}")
        print("Please train the model first using train.py")
        return None, None
    
    model.eval()
    
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return None, None
    
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None, None
    
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
    
    features = mel_spectrogram(waveform).squeeze(0).transpose(0, 1).to(device)
    features = features.unsqueeze(0)
    
    prompt_ids = None
    if prompt:
        prompt_map = {
            'punctuate': torch.tensor([0], dtype=torch.long, device=device),
            'grammar': torch.tensor([1], dtype=torch.long, device=device),
            'format': torch.tensor([2], dtype=torch.long, device=device)
        }
        prompt_ids = prompt_map.get(prompt.lower(), None)
        if prompt_ids is not None:
            prompt_ids = prompt_ids.unsqueeze(0)
    
    with torch.no_grad():
        decoded_tokens, disfluency_logits = inference(
            model, features, prompt_ids, max_length=512
        )
    
    idx_to_char = get_idx_to_char()
    
    transcription = decode_tokens(decoded_tokens[0], idx_to_char)
    
    disfluency_analysis = analyze_disfluency(disfluency_logits[0])
    
    return transcription, disfluency_analysis

def decode_tokens(tokens, idx_to_char):
    text = ''
    for token in tokens:
        if token.item() in idx_to_char and token.item() != 0:
            text += idx_to_char[token.item()]
    return text.strip()

def analyze_disfluency(disfluency_logits):
    if disfluency_logits is None:
        return {
            'total_frames': 0,
            'silence_frames': 0,
            'disfluency_frames': 0,
            'speech_frames': 0,
            'disfluency_ratio': 0.0
        }
    
    predictions = torch.argmax(disfluency_logits, dim=-1)
    
    silence_count = (predictions == 0).sum().item()
    disfluency_count = (predictions == 1).sum().item()
    speech_count = (predictions == 2).sum().item()
    
    total_frames = predictions.size(0)
    
    analysis = {
        'total_frames': total_frames,
        'silence_frames': silence_count,
        'disfluency_frames': disfluency_count,
        'speech_frames': speech_count,
        'disfluency_ratio': disfluency_count / total_frames if total_frames > 0 else 0
    }
    
    return analysis

class StreamingTranscriber:
    def __init__(self, model_path, chunk_size=160, hop_length=80):
        config = {
            'input_dim': 80,
            'vocab_size': 1000,
            'encoder_dim': 512,
            'decoder_dim': 512
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = HybridSqueezeStreamingASR(
            input_dim=config['input_dim'],
            vocab_size=config['vocab_size'],
            encoder_dim=config['encoder_dim'],
            decoder_dim=config['decoder_dim']
        ).to(self.device)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        self.processor = StreamingProcessor(self.model, chunk_size, hop_length)
        self.idx_to_char = get_idx_to_char()
        
    def transcribe_stream(self, audio_chunks):
        transcriptions = []
        
        for chunk in audio_chunks:
            chunk_tensor = torch.tensor(chunk).float().to(self.device)
            encoder_output, disfluency_logits = self.processor.process_chunk(chunk_tensor)
            
            if encoder_output is not None:
                with torch.no_grad():
                    dummy_tokens = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                    decoder_output = self.model.decoder(dummy_tokens, encoder_output)
                    joint_logits = self.model.joint_network(encoder_output, decoder_output)
                    
                    predicted_tokens = torch.argmax(joint_logits, dim=-1)
                    text = decode_tokens(predicted_tokens[0, :, 0], self.idx_to_char)
                    
                    if text.strip():
                        transcriptions.append(text)
        
        return ' '.join(transcriptions)

def batch_transcribe(audio_dir, output_file, config_path, prompt=None):
    audio_files = []
    for ext in ['*.wav', '*.flac', '*.mp3', '*.m4a']:
        audio_files.extend(list(Path(audio_dir).glob(ext)))
    
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return
    
    results = []
    
    for audio_file in audio_files:
        print(f"Transcribing: {audio_file}")
        
        try:
            transcription, disfluency_analysis = transcribe_audio(str(audio_file), config_path, prompt)
            
            if transcription is not None:
                result = {
                    'file': str(audio_file),
                    'transcription': transcription,
                    'disfluency_analysis': disfluency_analysis
                }
                results.append(result)
                print(f"  -> {transcription}")
            else:
                print(f"  -> Failed to transcribe")
                
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Batch transcription complete. Results saved to {output_file}")

def real_time_transcribe(config_path, duration=10):
    try:
        import pyaudio
        import numpy as np
    except ImportError:
        print("pyaudio is required for real-time transcription")
        print("Install with: pip install pyaudio")
        return
    
    transcriber = StreamingTranscriber(config_path)
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        print(f"Recording for {duration} seconds...")
        
        audio_chunks = []
        for _ in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_chunks.append(audio_chunk)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        transcription = transcriber.transcribe_stream(audio_chunks)
        print(f"Transcription: {transcription}")
        
        return transcription
        
    except Exception as e:
        print(f"Error during real-time transcription: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Transcribe audio with Hybrid Squeeze-Streaming ASR')
    parser.add_argument('--audio', type=str, help='Path to audio file')
    parser.add_argument('--config', type=str, default='eval_config.yaml', help='Path to config file')
    parser.add_argument('--prompt', type=str, choices=['punctuate', 'grammar', 'format'], help='Transcription prompt')
    parser.add_argument('--batch_dir', type=str, help='Directory containing audio files for batch processing')
    parser.add_argument('--output', type=str, default='transcriptions.json', help='Output file for batch processing')
    parser.add_argument('--streaming', action='store_true', help='Enable streaming transcription')
    parser.add_argument('--duration', type=int, default=10, help='Duration for streaming (seconds)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file {args.config} not found. Creating default config...")
        from config import create_configs
        create_configs()
    
    if args.streaming:
        real_time_transcribe(args.config, args.duration)
    elif args.batch_dir:
        batch_transcribe(args.batch_dir, args.output, args.config, args.prompt)
    elif args.audio:
        transcription, analysis = transcribe_audio(args.audio, args.config, args.prompt)
        
        if transcription is not None:
            print(f"Transcription: {transcription}")
            print("\nDisfluency Analysis:")
            print(f"  Total frames: {analysis['total_frames']}")
            print(f"  Disfluency frames: {analysis['disfluency_frames']}")
            print(f"  Disfluency ratio: {analysis['disfluency_ratio']:.3f}")
            
            if analysis['disfluency_ratio'] > 0.1:
                print("  Suggestion: Consider speaking more clearly to reduce disfluencies")
        else:
            print("Transcription failed")
    else:
        print("Please provide either --audio for single file or --batch_dir for batch processing")
        print("Or use --streaming for real-time transcription")

if __name__ == "__main__":
    main()