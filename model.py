"""
Laboratório 04 - O Transformer Completo From Scratch
Disciplina: Tópicos em Inteligência Artificial – 2026.1

EncoderBlock(x)
  1. Self-Attention (sem máscara)        → Add & Norm
  2. FFN                                 → Add & Norm
  → produz Z: memória rica contextualizada

DecoderBlock(y, Z)
  1. Masked Self-Attention (máscara causal) → Add & Norm
  2. Cross-Attention (Q=Decoder, K/V=Z)     → Add & Norm
  3. FFN                                    → Add & Norm
  → projeção linear + Softmax  →  P(vocab)
"""

import numpy as np
from components import (
    scaled_dot_product_attention,
    FeedForwardNetwork,
    LayerNorm,
    add_and_norm,
    create_causal_mask,
    D_MODEL, D_K, D_V, D_FF,
)

VOCAB_SIZE = 10_000

# PROJEÇÕES DE ATENÇÃO

def _make_projections(d_model: int = D_MODEL,
                      d_k: int = D_K,
                      d_v: int = D_V):

    return (
        np.random.randn(d_model, d_k) * 0.01,
        np.random.randn(d_model, d_k) * 0.01,
        np.random.randn(d_model, d_v) * 0.01,
    )


# TAREFA 2 — ENCODER BLOCK

class EncoderBlock:
    """
    Uma camada do Encoder

    Fluxo:
        x → Self-Attention → Add & Norm → FFN → Add & Norm → Z
    """

    def __init__(self, d_model: int = D_MODEL,
                 d_k: int = D_K, d_v: int = D_V, d_ff: int = D_FF):
        # Pesos de self-attention + projeção de saída d_v -> d_model
        self.W_Q, self.W_K, self.W_V = _make_projections(d_model, d_k, d_v)
        self.W_O = np.random.randn(d_v, d_model) * 0.01

        # Sub-camadas
        self.ffn  = FeedForwardNetwork(d_model, d_ff)
        self.ln1  = LayerNorm(d_model)
        self.ln2  = LayerNorm(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x : (batch, seq_src, d_model)
        Z : (batch, seq_src, d_model)
        """
        # Passo 1 — Self-Attention (sem máscara: Encoder vê toda a sequência)
        Q, K, V       = x @ self.W_Q, x @ self.W_K, x @ self.W_V
        att_out, _    = scaled_dot_product_attention(Q, K, V, mask=None)
        att_out       = att_out @ self.W_O   # projeta d_v -> d_model
        x             = add_and_norm(x, att_out, self.ln1)

        # Passo 2 — FFN
        ffn_out = self.ffn.forward(x)
        Z       = add_and_norm(x, ffn_out, self.ln2)

        return Z


class EncoderStack:
    """Pilha de N EncoderBlocks empilháveis."""

    def __init__(self, n_layers: int = 6, **kwargs):
        self.layers = [EncoderBlock(**kwargs) for _ in range(n_layers)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x   # Z: memória rica


# TAREFA 3 — DECODER BLOCK

class DecoderBlock:
    """
    Uma camada do Decoder.

    Fluxo:
        y → Masked Self-Attention → Add & Norm
          → Cross-Attention(Z)    → Add & Norm
          → FFN                   → Add & Norm
          → Linear → Softmax      → P(vocab)
    """

    def __init__(self, d_model: int = D_MODEL,
                 d_k: int = D_K, d_v: int = D_V, d_ff: int = D_FF,
                 vocab_size: int = VOCAB_SIZE):
        # Pesos de masked self-attention + projecao de saida
        self.W_Q1, self.W_K1, self.W_V1 = _make_projections(d_model, d_k, d_v)
        self.W_O1 = np.random.randn(d_v, d_model) * 0.01

        # Pesos de cross-attention + projecao de saida
        self.W_Q2, self.W_K2, self.W_V2 = _make_projections(d_model, d_k, d_v)
        self.W_O2 = np.random.randn(d_v, d_model) * 0.01

        # Sub-camadas
        self.ffn  = FeedForwardNetwork(d_model, d_ff)
        self.ln1  = LayerNorm(d_model)
        self.ln2  = LayerNorm(d_model)
        self.ln3  = LayerNorm(d_model)

        # Projeção final: d_model → vocab_size
        self.W_out = np.random.randn(d_model, vocab_size) * 0.01

    def forward(self, y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        y : (batch, seq_tgt, d_model)  — tokens do Decoder até agora
        Z : (batch, seq_src, d_model)  — memória rica do Encoder

        Retorna
        -------
        probs : (batch, seq_tgt, vocab_size)
        """
        seq_tgt = y.shape[1]

        # Passo 1 — Masked Self-Attention (causal: não olha para o futuro)
        mask          = create_causal_mask(seq_tgt)
        Q1, K1, V1    = y @ self.W_Q1, y @ self.W_K1, y @ self.W_V1
        att1_out, _   = scaled_dot_product_attention(Q1, K1, V1, mask=mask)
        att1_out      = att1_out @ self.W_O1
        y             = add_and_norm(y, att1_out, self.ln1)

        # Passo 2 — Cross-Attention (Q do Decoder, K/V da memória Z)
        Q2, K2, V2    = y @ self.W_Q2, Z @ self.W_K2, Z @ self.W_V2
        att2_out, _   = scaled_dot_product_attention(Q2, K2, V2, mask=None)
        att2_out      = att2_out @ self.W_O2
        y             = add_and_norm(y, att2_out, self.ln2)

        # Passo 3 — FFN
        ffn_out = self.ffn.forward(y)
        y       = add_and_norm(y, ffn_out, self.ln3)

        # Projeção linear → vocab + Softmax → distribuição de probabilidades
        logits = y @ self.W_out                          # (batch, seq_tgt, vocab)
        exp_l  = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs  = exp_l / exp_l.sum(axis=-1, keepdims=True)
        return probs


class DecoderStack:
    """Pilha de N DecoderBlocks."""

    def __init__(self, n_layers: int = 6, **kwargs):
        self.layers = [DecoderBlock(**kwargs) for _ in range(n_layers)]

    def forward(self, y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        probs = None
        for layer in self.layers:
            probs = layer.forward(y, Z)
            # Alimenta a saída normalizada de volta (sem re-embeddar)
        return probs


# SMOKE TEST

if __name__ == "__main__":
    np.random.seed(42)
    BATCH, SEQ_SRC, SEQ_TGT = 1, 6, 4

    X = np.random.randn(BATCH, SEQ_SRC, D_MODEL)
    Y = np.random.randn(BATCH, SEQ_TGT, D_MODEL)

    # Encoder
    encoder = EncoderStack(n_layers=6)
    Z       = encoder.forward(X)
    print(f"[EncoderStack] Z shape: {Z.shape}  (esperado: ({BATCH},{SEQ_SRC},{D_MODEL}))")

    # Decoder (1 bloco para smoke test rápido)
    decoder = DecoderBlock()
    probs   = decoder.forward(Y, Z)
    print(f"[DecoderBlock] probs shape: {probs.shape}  (esperado: ({BATCH},{SEQ_TGT},{VOCAB_SIZE}))")

    soma_ok = np.allclose(probs[0].sum(axis=-1), 1.0)
    print(f"  Probs somam 1.0 por token? {'✓ SIM' if soma_ok else '✗ NÃO'}")

    print("\n[Tarefas 2 e 3 concluídas ✓]")
