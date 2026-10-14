"""The one model configuration used by every run (pre-training and both fine-tuning conditions):
T5-small (6 + 6 layers, 8 heads, d_model 512, feed-forward 2,048) with our 32,100-token vocabulary.
T5Config's other defaults apply (dropout 0.1, ReLU feed-forward, tied embeddings)."""
MODEL_CONFIG = dict(vocab_size=32100, d_model=512, d_ff=2048, num_layers=6, num_decoder_layers=6, num_heads=8,
                    d_kv=64, pad_token_id=0, eos_token_id=1, decoder_start_token_id=0)
