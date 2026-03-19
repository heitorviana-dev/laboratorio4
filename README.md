# Transformer Completo — From Scratch

**Disciplina:** Tópicos em Inteligência Artificial – 2026.1  
**Professor:** Prof. Dimmy Magalhães  
**Instituição:** iCEV - Instituto de Ensino Superior

Implementação da arquitetura Encoder-Decoder completa do Transformer,
baseada em *"Attention Is All You Need"* (Vaswani et al., 2017),
usando apenas `Python 3` e `numpy`.

---

## Estrutura do Projeto

```
transformer-complete/
├── components.py   # Tarefa 1 — Attention, FFN, LayerNorm, Add & Norm
├── model.py        # Tarefas 2 e 3 — EncoderBlock, DecoderBlock e Stacks
├── inference.py    # Tarefa 4 — Pipeline fim-a-fim com loop auto-regressivo
└── README.md
```

---

## Como Rodar

### Pré-requisitos

```bash
pip install numpy
```

### Executar

```bash
# Testa os blocos individualmente
python components.py

# Testa Encoder e Decoder isoladamente
python model.py

# Roda o pipeline completo de inferência
python inference.py
```

---

## Arquitetura

```
"Thinking Machines"
       │
  Embedding + PE
       │
  ┌────▼────────────────────┐
  │  EncoderBlock × 6       │
  │  Self-Attention         │
  │  Add & Norm             │
  │  FFN                    │
  │  Add & Norm             │
  └────────────┬────────────┘
               │  Z (memória rica)
  ┌────────────▼────────────┐
  │  DecoderBlock × 6       │
  │  Masked Self-Attention  │◄── <START> ... tokens gerados
  │  Add & Norm             │
  │  Cross-Attention (Z)    │
  │  Add & Norm             │
  │  FFN                    │
  │  Add & Norm             │
  │  Linear → Softmax       │
  └────────────┬────────────┘
               │  P(vocab)  →  argmax  →  próximo token
               └──────────────────────────► loop até <EOS>
```

---

## Fluxo de Tensores

| Etapa | Shape |
|---|---|
| Entrada Encoder | `(1, seq_src, 512)` |
| Memória Z | `(1, seq_src, 512)` |
| Entrada Decoder | `(1, seq_tgt, 512)` |
| Saída Decoder (probs) | `(1, seq_tgt, 10000)` |

---

## Nota de Integridade Acadêmica

Claude (Anthropic) foi consultado como ferramenta auxiliar para revisão de
estrutura de código e sintaxe NumPy, conforme permitido pelo Contrato Pedagógico.
A lógica, implementação matemática e decisões de arquitetura são de autoria do aluno Heitor Viana.
