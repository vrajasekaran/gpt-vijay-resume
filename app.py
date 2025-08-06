import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import json
from collections import Counter
import math

import re
from collections import Counter
import math


# ==========================================
# MINI-GPT MODEL ARCHITECTURE
# ==========================================

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal mask (for autoregressive generation)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        return self.out_proj(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_out = self.attention(self.ln1(x))
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_out)
        
        return x


class VijayResumeGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=6, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Output
        self.ln_final = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = token_emb + pos_emb
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Final layer norm and projection
        x = self.ln_final(x)
        logits = self.output_projection(x)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    pass



    
# ==========================================
# TOKENIZER
# ==========================================

class SimpleTokenizer:
    def __init__(self, texts):
        self.special_tokens = {
            '<|pad|>': 0,
            '<|start|>': 1,
            '<|end|>': 2,
            '<|unk|>': 3
        }
        
        # Create vocabulary from training texts
        all_text = ' '.join(texts)
        words = re.findall(r'\w+|[^\w\s]', all_text.lower())
        word_counts = Counter(words)
        
        # Keep most common words
        vocab_words = [word for word, count in word_counts.most_common(5000)]
        
        # Build vocabulary
        self.vocab = self.special_tokens.copy()
        for word in vocab_words:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        print(f"Vocabulary size: {self.vocab_size}")
    
    def encode(self, text):
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        return [self.vocab.get(word, self.vocab['<|unk|>']) for word in words]
    
    def decode(self, tokens):
        return ' '.join([self.reverse_vocab.get(token, '<|unk|>') for token in tokens])
    
    def encode_batch(self, texts, max_length=512):
        encoded_texts = []
        for text in texts:
            encoded = self.encode(text)
            # Truncate or pad
            if len(encoded) > max_length:
                encoded = encoded[:max_length]
            else:
                encoded = encoded + [self.vocab['<|pad|>']] * (max_length - len(encoded))
            encoded_texts.append(encoded)
        return torch.tensor(encoded_texts, dtype=torch.long)

    pass

# Load the trained model
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load('model_files/vijay_resume_gpt_final.pt', map_location=device)
    
    # Recreate tokenizer
    vocab = checkpoint['tokenizer_vocab']
    reverse_vocab = {v: k for k, v in vocab.items()}
    tokenizer = SimpleTokenizer([])  # Initialize empty, then load vocab
    tokenizer.vocab = vocab
    tokenizer.reverse_vocab = reverse_vocab
    tokenizer.vocab_size = len(vocab)
    
    # Recreate model
    config = checkpoint['model_config']
    model = VijayResumeGPT(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer

model, tokenizer = load_model()

def generate_response(prompt, max_length=150, temperature=0.7):
    """Generate response from Vijay's Resume GPT"""
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Format prompt
        formatted_prompt = f"<|start|>Human: {prompt}\nVijay:"
        
        # Encode prompt
        input_ids = torch.tensor([tokenizer.encode(formatted_prompt)], dtype=torch.long).to(device)
        
        # Generate tokens
        for _ in range(max_length):
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            if next_token.item() == tokenizer.vocab.get('<|end|>', -1):
                break
        
        # Decode response
        generated_text = tokenizer.decode(input_ids[0].cpu().tolist())
        
        # Extract Vijay's response
        try:
            response = generated_text.split("Vijay:")[-1].split("<|end|>")[0].strip()
            return response
        except:
            return "I'm having trouble generating a response. Please try rephrasing your question."

def chat_interface(message, history):
    """Gradio chat interface"""
    if not message.strip():
        return "", history
    
    # Generate response
    response = generate_response(message, temperature=0.8)
    
    # Update history
    history.append([message, response])
    
    return "", history

# Create Gradio interface
with gr.Blocks(title="Chat with Vijay Rajasekaran's Resume", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 Chat with Vijay Rajasekaran's Resume GPT
    
    Ask me anything about Vijay's professional background, experience, skills, or career journey!
    This AI model was trained exclusively on Vijay's resume data.
    
    **Sample questions:**
    - "What's your current role?"
    - "Tell me about your AI experience"
    - "What programming languages do you know?"
    - "What did you build at EquiB?"
    """)
    
    chatbot = gr.Chatbot(
        value=[],
        height=500,
        bubble_full_width=False,
        avatar_images=["👤", "🤖"]
    )
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask me about Vijay's experience...",
            container=False,
            scale=7
        )
        submit = gr.Button("Send", scale=1, variant="primary")
    
    with gr.Row():
        clear = gr.Button("Clear Chat")
    
    gr.Examples(
        examples=[
            "What's your name and current role?",
            "Tell me about your AI and machine learning experience",
            "What did you build at EquiB?",
            "What programming languages and frameworks do you use?",
            "What's your experience with RAG and LLM systems?",
            "How can I contact you?"
        ],
        inputs=msg
    )
    
    # Event handlers
    msg.submit(chat_interface, [msg, chatbot], [msg, chatbot])
    submit.click(chat_interface, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: ([], ""), outputs=[chatbot, msg])

if __name__ == "__main__":
    demo.launch()