"""
Laboratório 04 - O Transformer Completo From Scratch
Disciplina: Tópicos em Inteligência Artificial

Pipeline completo:
  1. Encoder recebe "Thinking Machines" e produz Z
  2. Decoder inicia com <START> e gera tokens auto-regressivamente
  3. Loop para ao gerar <EOS>
"""

import numpy as np
from model import EncoderStack, DecoderBlock, VOCAB_SIZE
from components import D_MODEL

np.random.seed(42)

# VOCABULÁRIO FICTÍCIO

START_TOKEN = "<START>"
EOS_TOKEN   = "<EOS>"
EOS_IDX     = VOCAB_SIZE - 2   # índice 9998

vocab = {i: f"word_{i}" for i in range(VOCAB_SIZE)}
vocab[0]       = START_TOKEN
vocab[EOS_IDX] = EOS_TOKEN

# Tabela de embeddings: idx → vetor d_model
embedding_table = np.random.randn(VOCAB_SIZE, D_MODEL) * 0.01

MAX_STEPS = 10

# HELPERS

def tokens_to_tensor(token_ids: list) -> np.ndarray:
    """Converte lista de IDs em tensor (1, seq_len, d_model)."""
    vecs = embedding_table[token_ids]      # (seq, d_model)
    return vecs[np.newaxis, :, :]          # (1, seq, d_model)


def encode_sentence(words: list, word_to_id: dict) -> np.ndarray:
    """Mapeia palavras para IDs e devolve tensor do Encoder."""
    ids = [word_to_id.get(w, 1) for w in words]   # 1 = <UNK>
    return tokens_to_tensor(ids)


# TAREFA 4 — INFERÊNCIA

def run_inference():
    print("=" * 58)
    print("TRANSFORMER COMPLETO — INFERÊNCIA FIM-A-FIM")
    print("=" * 58)

    encoder = EncoderStack(n_layers=6)
    decoder = DecoderBlock()

    frase_entrada = ["Thinking", "Machines"]
    word_to_id    = {w: i + 10 for i, w in enumerate(frase_entrada)}

    encoder_input = encode_sentence(frase_entrada, word_to_id)
    print(f"\nEntrada Encoder : {frase_entrada}")
    print(f"encoder_input shape: {encoder_input.shape}")
    
    Z = encoder.forward(encoder_input)
    print(f"Memória Z shape    : {Z.shape}")

    print(f"\n{'─'*58}")
    print("LOOP AUTO-REGRESSIVO")
    print(f"{'─'*58}")

    generated_ids = [0]   # começa com índice do <START>
    print(f"  Início: [{START_TOKEN}]")

    step = 0
    while step < MAX_STEPS:
        # Monta tensor com toda a sequência gerada até agora
        decoder_input = tokens_to_tensor(generated_ids)  # (1, seq, d_model)

        # Forward pass do Decoder → (1, seq, vocab_size)
        probs = decoder.forward(decoder_input, Z)

        # Toma as probabilidades do ÚLTIMO token (prediz o próximo)
        next_probs = probs[0, -1, :]                     # (vocab_size,)

        # Força EOS no passo 5 para demonstrar a parada do loop
        # (em modelo treinado o EOS emerge naturalmente)
        if step >= 4:
            next_probs[EOS_IDX] = next_probs.max() + 1.0
            next_probs = np.exp(next_probs) / np.exp(next_probs).sum()

        next_idx   = int(np.argmax(next_probs))
        next_token = vocab[next_idx]

        generated_ids.append(next_idx)
        step += 1

        print(f"  Passo {step:02d} | idx={next_idx:5d} | "
              f"token='{next_token}' | prob={next_probs[next_idx]:.4f}")

        if next_token == EOS_TOKEN:
            print(f"\n  <EOS> detectado — geração encerrada.")
            break

    frase_final = [vocab[i] for i in generated_ids if vocab[i] != START_TOKEN]
    print(f"\n{'='*58}")
    print("FRASE GERADA (sem <START>):")
    print(f"  {' '.join(frase_final)}")
    print(f"  ({len(frase_final)} tokens)")
    print(f"{'='*58}")
    print("\n[Tarefa 4 concluída ✓  —  Pipeline fim-a-fim funcionando!]")


if __name__ == "__main__":
    run_inference()
