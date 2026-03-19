"""
Laboratório 04 - O Transformer Completo From Scratch
Disciplina: Tópicos em Inteligência Artificial

Consolida em um único módulo os três blocos fundamentais
usados por Encoder e Decoder:

  - scaled_dot_product_attention(Q, K, V, mask)
  - FeedForwardNetwork
  - add_and_norm(x, sublayer_output, layer_norm)
"""

import numpy as np

# HIPERPARÂMETROS GLOBAIS
D_MODEL = 512
D_K     = 64
D_V     = 64
D_FF    = 2048
EPSILON = 1e-6

# UTILITÁRIOS

def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax numericamente estável no último eixo."""
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Matriz [seq_len, seq_len]:
      triângulo inferior + diagonal  →  0
      triângulo superior             →  -inf
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return np.where(mask == 1, -np.inf, 0.0)


# BLOCO 1 — SCALED DOT-PRODUCT ATTENTION

def scaled_dot_product_attention(Q: np.ndarray,
                                 K: np.ndarray,
                                 V: np.ndarray,
                                 mask: np.ndarray = None) -> np.ndarray:

    d_k    = Q.shape[-1]
    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_k) 

    if mask is not None:
        scores = scores + mask

    weights = softmax(scores)
    output  = weights @ V     
    return output, weights

# BLOCO 2 — FEED-FORWARD NETWORK

class FeedForwardNetwork:

    def __init__(self, d_model: int = D_MODEL, d_ff: int = D_FF):
        self.W1 = np.random.randn(d_model, d_ff)  * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff,   d_model) * 0.01
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x : (batch, seq, d_model)  →  (batch, seq, d_model)"""
        hidden = np.maximum(0, x @ self.W1 + self.b1)
        return hidden @ self.W2 + self.b2


# BLOCO 3 — ADD & NORM

class LayerNorm:
    """Normalização de camada no último eixo (features)."""

    def __init__(self, d_model: int = D_MODEL, epsilon: float = EPSILON):
        self.epsilon = epsilon
        self.gamma   = np.ones(d_model)
        self.beta    = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean  = np.mean(x, axis=-1, keepdims=True)
        var   = np.var (x, axis=-1, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + self.epsilon)
        return self.gamma * x_hat + self.beta


def add_and_norm(x: np.ndarray,
                 sublayer_output: np.ndarray,
                 layer_norm: LayerNorm) -> np.ndarray:
    """
    Output = LayerNorm(x + Sublayer(x))

    Conexão residual seguida de normalização de camada.
    """
    return layer_norm.forward(x + sublayer_output)


# SMOKE TEST

if __name__ == "__main__":
    np.random.seed(42)
    BATCH, SEQ = 1, 5

    X = np.random.randn(BATCH, SEQ, D_MODEL)

    # Projeta Q, K, V a partir de X para testar a attention
    W_Q = np.random.randn(D_MODEL, D_K) * 0.01
    W_K = np.random.randn(D_MODEL, D_K) * 0.01
    W_V = np.random.randn(D_MODEL, D_V) * 0.01
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V

    mask        = create_causal_mask(SEQ)
    att_out, _  = scaled_dot_product_attention(Q, K, V, mask)
    print(f"[Attention]  shape: {att_out.shape}")

    ffn     = FeedForwardNetwork()
    ffn_out = ffn.forward(X)
    print(f"[FFN]        shape: {ffn_out.shape}")

    ln      = LayerNorm()
    norm    = add_and_norm(X, ffn_out, ln)
    print(f"[Add & Norm] shape: {norm.shape}")

    print("\n[Tarefa 1 concluída ✓]")
