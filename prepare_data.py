import json
import os
import argparse
from pathlib import Path

def create_data_manifest(audio_dir, transcript_file, output_manifest):
    manifest_data = []
    
    with open(transcript_file, 'r') as f:
        transcripts = f.readlines()
    
    for i, transcript in enumerate(transcripts):
        audio_path = os.path.join(audio_dir, f"{i:06d}.wav")
        if os.path.exists(audio_path):
            manifest_entry = {
                "audio_filepath": audio_path,
                "text": transcript.strip(),
                "duration": 0.0,
                "lang": "en"
            }
            manifest_data.append(manifest_entry)
    
    with open(output_manifest, 'w') as f:
        for entry in manifest_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Created manifest with {len(manifest_data)} entries: {output_manifest}")

def prepare_librispeech(librispeech_root, output_dir):
    if not os.path.exists(librispeech_root):
        print(f"LibriSpeech directory not found: {librispeech_root}")
        print("Please download and extract LibriSpeech dataset first")
        return False
    
    splits = ['train-clean-100', 'train-clean-360', 'dev-clean', 'test-clean']
    
    for split in splits:
        split_dir = os.path.join(librispeech_root, split)
        if not os.path.exists(split_dir):
            print(f"Split directory not found: {split_dir}, skipping...")
            continue
            
        manifest_path = os.path.join(output_dir, f"{split}_manifest.json")
        manifest_data = []
        
        print(f"Processing {split}...")
        
        for speaker_dir in os.listdir(split_dir):
            speaker_path = os.path.join(split_dir, speaker_dir)
            if not os.path.isdir(speaker_path):
                continue
                
            for chapter_dir in os.listdir(speaker_path):
                chapter_path = os.path.join(speaker_path, chapter_dir)
                if not os.path.isdir(chapter_path):
                    continue
                
                transcript_file = os.path.join(chapter_path, f"{speaker_dir}-{chapter_dir}.trans.txt")
                if not os.path.exists(transcript_file):
                    continue
                
                with open(transcript_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        if len(parts) < 2:
                            continue
                        
                        file_id = parts[0]
                        text = parts[1]
                        audio_path = os.path.join(chapter_path, f"{file_id}.flac")
                        
                        if os.path.exists(audio_path):
                            manifest_entry = {
                                "audio_filepath": audio_path,
                                "text": text.lower(),
                                "duration": 0.0,
                                "speaker_id": speaker_dir,
                                "lang": "en"
                            }
                            manifest_data.append(manifest_entry)
        
        with open(manifest_path, 'w') as f:
            for entry in manifest_data:
                f.write(json.dumps(entry) + '\n')
        
        print(f"Created {split} manifest with {len(manifest_data)} entries: {manifest_path}")
    
    return True

def create_synthetic_prompt_data(base_manifest, output_manifest, num_prompts=1000):
    if not os.path.exists(base_manifest):
        print(f"Base manifest not found: {base_manifest}")
        return
    
    prompts = [
        "Add punctuation",
        "Correct grammar", 
        "Format as formal text",
        "Remove filler words",
        "Capitalize properly",
        "Add paragraph breaks",
        "Convert to bullet points",
        "Summarize content"
    ]
    
    with open(base_manifest, 'r') as f:
        base_data = [json.loads(line) for line in f if line.strip()]
    
    if not base_data:
        print("No data found in base manifest")
        return
    
    prompt_data = []
    for i in range(min(num_prompts, len(base_data) * len(prompts))):
        base_entry = base_data[i % len(base_data)]
        prompt = prompts[i % len(prompts)]
        
        prompt_entry = base_entry.copy()
        prompt_entry['prompt'] = prompt
        prompt_entry['prompt_id'] = i % len(prompts)
        
        prompt_data.append(prompt_entry)
    
    with open(output_manifest, 'w') as f:
        for entry in prompt_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Created prompt manifest with {len(prompt_data)} entries: {output_manifest}")

def augment_with_noise(clean_manifest, noise_dir, output_manifest, snr_levels=[20, 15, 10, 5]):
    if not os.path.exists(clean_manifest):
        print(f"Clean manifest not found: {clean_manifest}")
        return
    
    if not os.path.exists(noise_dir):
        print(f"Noise directory not found: {noise_dir}")
        return
    
    with open(clean_manifest, 'r') as f:
        clean_data = [json.loads(line) for line in f if line.strip()]
    
    noise_files = [f for f in os.listdir(noise_dir) if f.endswith('.wav')]
    if not noise_files:
        print("No noise files found")
        return
    
    augmented_data = []
    
    for entry in clean_data[:100]:  # Limit to first 100 for demo
        for snr in snr_levels:
            for noise_file in noise_files[:3]:
                aug_entry = entry.copy()
                aug_entry['noise_file'] = os.path.join(noise_dir, noise_file)
                aug_entry['snr'] = snr
                aug_entry['augmented'] = True
                augmented_data.append(aug_entry)
    
    with open(output_manifest, 'w') as f:
        for entry in augmented_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Created augmented manifest with {len(augmented_data)} entries: {output_manifest}")

def create_disfluency_labels(manifest_file, output_file):
    if not os.path.exists(manifest_file):
        print(f"Manifest file not found: {manifest_file}")
        return
    
    with open(manifest_file, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    disfluency_patterns = ['um', 'uh', 'er', 'ah', 'like', 'you know']
    
    for entry in data:
        text = entry['text'].lower()
        disfluency_count = sum(text.count(pattern) for pattern in disfluency_patterns)
        
        entry['has_disfluency'] = disfluency_count > 0
        entry['disfluency_count'] = disfluency_count
        
        words = text.split()
        word_labels = []
        for word in words:
            if word in disfluency_patterns:
                word_labels.append(1)  # disfluency
            else:
                word_labels.append(0)  # clean speech
        
        entry['word_disfluency_labels'] = word_labels
    
    with open(output_file, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Added disfluency labels to {len(data)} entries: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Data preparation for Hybrid Squeeze-Streaming ASR')
    parser.add_argument('--librispeech_root', type=str, help='Path to LibriSpeech dataset')
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory for manifests')
    parser.add_argument('--noise_dir', type=str, help='Path to noise files for augmentation')
    parser.add_argument('--create_prompts', action='store_true', help='Create synthetic prompt data')
    parser.add_argument('--add_noise', action='store_true', help='Add noise augmentation')
    parser.add_argument('--add_disfluency', action='store_true', help='Add disfluency labels')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    success = True
    
    if args.librispeech_root:
        success = prepare_librispeech(args.librispeech_root, args.output_dir)
    
    if success and args.create_prompts:
        base_manifest = os.path.join(args.output_dir, 'train-clean-100_manifest.json')
        prompt_manifest = os.path.join(args.output_dir, 'prompt_manifest.json')
        if os.path.exists(base_manifest):
            create_synthetic_prompt_data(base_manifest, prompt_manifest)
        else:
            print(f"Base manifest not found for prompts: {base_manifest}")
    
    if success and args.add_noise and args.noise_dir:
        clean_manifest = os.path.join(args.output_dir, 'train-clean-100_manifest.json')
        noisy_manifest = os.path.join(args.output_dir, 'train_noisy_manifest.json')
        if os.path.exists(clean_manifest):
            augment_with_noise(clean_manifest, args.noise_dir, noisy_manifest)
        else:
            print(f"Clean manifest not found for noise augmentation: {clean_manifest}")
    
    if success and args.add_disfluency:
        for split in ['train-clean-100', 'dev-clean', 'test-clean']:
            manifest_file = os.path.join(args.output_dir, f'{split}_manifest.json')
            output_file = os.path.join(args.output_dir, f'{split}_disfluency_manifest.json')
            if os.path.exists(manifest_file):
                create_disfluency_labels(manifest_file, output_file)
            else:
                print(f"Manifest not found: {manifest_file}")
    
    if not success:
        print("\nData preparation failed. Please check the LibriSpeech path and try again.")
        print("\nQuick start:")
        print("1. Download LibriSpeech: https://www.openslr.org/12/")
        print("2. Extract to a directory (e.g., /path/to/LibriSpeech/)")
        print("3. Run: python prepare_data.py --librispeech_root /path/to/LibriSpeech")

if __name__ == "__main__":
    main()