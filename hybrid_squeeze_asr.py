import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TemporalUNet(nn.Module):
    def __init__(self, d_model, num_layers=12):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        
        for i in range(num_layers // 2):
            self.downsample_layers.append(
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
                    nn.LayerNorm(d_model),
                    nn.SiLU()
                )
            )
        
        for i in range(num_layers // 2):
            self.upsample_layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(d_model, d_model, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.LayerNorm(d_model),
                    nn.SiLU()
                )
            )
    
    def forward(self, x):
        skip_connections = []
        
        for downsample in self.downsample_layers:
            skip_connections.append(x)
            x = x.transpose(1, 2)
            x = downsample(x)
            x = x.transpose(1, 2)
        
        for upsample in self.upsample_layers:
            x = x.transpose(1, 2)
            x = upsample(x)
            x = x.transpose(1, 2)
            if skip_connections:
                skip = skip_connections.pop()
                if x.shape[1] != skip.shape[1]:
                    x = F.interpolate(x.transpose(1, 2), size=skip.shape[1], mode='linear').transpose(1, 2)
                x = x + skip
        
        return x

class SqueezeFormerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.conv1d = nn.Conv1d(d_model, dim_feedforward, kernel_size=3, padding=1)
        self.conv1d_out = nn.Conv1d(dim_feedforward, d_model, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()
        
    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = residual + self.dropout(attn_out)
        
        residual = x
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = self.activation(x)
        x = self.conv1d_out(x)
        x = x.transpose(1, 2)
        x = residual + self.dropout(x)
        
        x = self.norm3(x)
        return x

class SqueezeFormerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=512, nhead=8, num_layers=12, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.temporal_unet = TemporalUNet(d_model, num_layers)
        
        self.blocks = nn.ModuleList([
            SqueezeFormerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.temporal_unet(x)
        x = self.final_norm(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class PromptConditionedLLMDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.prompt_embedding = nn.Embedding(100, d_model)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, prompt_ids=None):
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        
        if prompt_ids is not None:
            prompt_embeds = self.prompt_embedding(prompt_ids)
            tgt = torch.cat([prompt_embeds, tgt], dim=1)
        
        tgt = self.pos_encoding(tgt)
        
        output = self.transformer_decoder(
            tgt, memory, 
            tgt_mask=tgt_mask, 
            memory_mask=memory_mask
        )
        
        return self.output_projection(output)

class TransducerJointNetwork(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, vocab_size, joint_dim=640):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, joint_dim)
        self.decoder_proj = nn.Linear(decoder_dim, joint_dim)
        self.joint_proj = nn.Linear(joint_dim, vocab_size)
        self.activation = nn.Tanh()
        
    def forward(self, encoder_outputs, decoder_outputs):
        enc_proj = self.encoder_proj(encoder_outputs)
        dec_proj = self.decoder_proj(decoder_outputs)
        
        B, T, _ = enc_proj.shape
        B, U, _ = dec_proj.shape
        
        enc_proj = enc_proj.unsqueeze(2).expand(B, T, U, -1)
        dec_proj = dec_proj.unsqueeze(1).expand(B, T, U, -1)
        
        joint = enc_proj + dec_proj
        joint = self.activation(joint)
        logits = self.joint_proj(joint)
        
        return logits

class DisfluencyDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3)
        )
        
    def forward(self, features):
        return self.classifier(features)

class HybridSqueezeStreamingASR(nn.Module):
    def __init__(self, 
                 input_dim=80, 
                 vocab_size=1000, 
                 encoder_dim=512, 
                 decoder_dim=512,
                 nhead=8,
                 encoder_layers=12,
                 decoder_layers=6,
                 dropout=0.1):
        super().__init__()
        
        self.encoder = SqueezeFormerEncoder(
            input_dim=input_dim,
            d_model=encoder_dim,
            nhead=nhead,
            num_layers=encoder_layers,
            dropout=dropout
        )
        
        self.decoder = PromptConditionedLLMDecoder(
            vocab_size=vocab_size,
            d_model=decoder_dim,
            nhead=nhead,
            num_layers=decoder_layers,
            dropout=dropout
        )
        
        self.joint_network = TransducerJointNetwork(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            vocab_size=vocab_size
        )
        
        self.disfluency_detector = DisfluencyDetector(encoder_dim)
        
    def forward(self, audio_features, text_tokens=None, prompt_ids=None):
        encoder_outputs = self.encoder(audio_features)
        
        disfluency_logits = self.disfluency_detector(encoder_outputs)
        
        if text_tokens is not None:
            decoder_outputs = self.decoder(
                text_tokens, 
                encoder_outputs,
                prompt_ids=prompt_ids
            )
            
            joint_logits = self.joint_network(encoder_outputs, decoder_outputs)
            
            return {
                'encoder_outputs': encoder_outputs,
                'decoder_outputs': decoder_outputs,
                'joint_logits': joint_logits,
                'disfluency_logits': disfluency_logits
            }
        else:
            return {
                'encoder_outputs': encoder_outputs,
                'disfluency_logits': disfluency_logits
            }

class RNNTLoss(nn.Module):
    def __init__(self, blank_idx=0):
        super().__init__()
        self.blank_idx = blank_idx
        
    def forward(self, logits, targets, input_lengths, target_lengths):
        try:
            from warp_rnnt import rnnt_loss
            return rnnt_loss(logits, targets, input_lengths, target_lengths, 
                           blank=self.blank_idx, reduction='mean')
        except ImportError:
            print("Warning: Using CTC loss as RNN-T fallback. Install warp_rnnt for proper RNN-T loss.")
            log_probs = F.log_softmax(logits.view(-1, logits.size(-1)), dim=-1)
            return F.ctc_loss(
                log_probs.view(logits.size(0), logits.size(1), -1).transpose(0, 1), 
                targets, 
                input_lengths, 
                target_lengths,
                blank=self.blank_idx,
                reduction='mean'
            )

def inference(model, audio_features, prompt_ids=None, max_length=512):
    model.eval()
    
    with torch.no_grad():
        outputs = model(audio_features, None, prompt_ids)
        encoder_outputs = outputs['encoder_outputs']
        disfluency_logits = outputs['disfluency_logits']
        
        batch_size = audio_features.size(0)
        device = audio_features.device
        
        decoded_tokens = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        for step in range(max_length):
            decoder_outputs = model.decoder(
                decoded_tokens,
                encoder_outputs,
                prompt_ids=prompt_ids
            )
            
            joint_logits = model.joint_network(encoder_outputs, decoder_outputs)
            
            next_token_logits = joint_logits[:, -1, :, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            decoded_tokens = torch.cat([decoded_tokens, next_token.unsqueeze(-1)], dim=1)
            
            if torch.all(next_token == 0):
                break
                
        return decoded_tokens, disfluency_logits

class StreamingProcessor:
    def __init__(self, model, chunk_size=160, hop_length=80):
        self.model = model
        self.chunk_size = chunk_size
        self.hop_length = hop_length
        self.cache = []
        self.model.eval()
        
    def process_chunk(self, audio_chunk):
        self.cache.append(audio_chunk)
        if len(self.cache) > self.chunk_size:
            self.cache.pop(0)
            
        if len(self.cache) >= self.hop_length:
            chunk_tensor = torch.cat(self.cache, dim=0).unsqueeze(0)
            with torch.no_grad():
                outputs = self.model(chunk_tensor)
                return outputs['encoder_outputs'], outputs['disfluency_logits']
        else:
            return None, None

def create_model(config):
    return HybridSqueezeStreamingASR(
        input_dim=config.get('input_dim', 80),
        vocab_size=config.get('vocab_size', 1000),
        encoder_dim=config.get('encoder_dim', 512),
        decoder_dim=config.get('decoder_dim', 512),
        nhead=config.get('nhead', 8),
        encoder_layers=config.get('encoder_layers', 12),
        decoder_layers=config.get('decoder_layers', 6),
        dropout=config.get('dropout', 0.1)
    )