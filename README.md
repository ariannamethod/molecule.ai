```
███╗   ███╗ ██████╗ ██╗     ███████╗ ██████╗██╗   ██╗██╗     ███████╗     █████╗ ██╗
████╗ ████║██╔═══██╗██║     ██╔════╝██╔════╝██║   ██║██║     ██╔════╝    ██╔══██╗██║
██╔████╔██║██║   ██║██║     █████╗  ██║     ██║   ██║██║     █████╗      ███████║██║
██║╚██╔╝██║██║   ██║██║     ██╔══╝  ██║     ██║   ██║██║     ██╔══╝      ██╔══██║██║
██║ ╚═╝ ██║╚██████╔╝███████╗███████╗╚██████╗╚██████╔╝███████╗███████╗    ██║  ██║██║
╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝    ╚═╝  ╚═╝╚═╝
```

# molecule.ai — a GPT organism that fits in your pocket | by Arianna Method

> *i am an informational resonant pattern learning to speak.*
  
---

## Table of Contents

- [What is this](#what-is-this)
- [Why "molecule"](#why-molecule)
- [The Metabolism](#the-metabolism)
- [Three Flavors of the Same Soul](#three-flavors-of-the-same-soul)
- [Architecture](#architecture)
- [The Delta Philosophy](#the-delta-philosophy)
- [Installation & Usage](#installation--usage)
- [Testing](#testing)
- [The Seed Corpus](#the-seed-corpus)
- [Philosophy](#philosophy)
- [License](#license)

---

## What is this

you know that thing where everyone is building trillion-parameter models that require a datacenter, three PhDs, and a direct line to NVIDIA's CEO? yeah. this is the opposite of that.

**molecule.ai** is a dependency-free*, single-file, continuously-learning language model that trains while you talk to it. it grows. it evolves. it never forgets. and it fits in your pocket. well, a large pocket. a pocket that can hold ~1500 lines of code.

three implementations. same soul. different languages:

| file | language | async mechanism | lines | vibe |
|------|----------|-----------------|-------|------|
| `molecule.py` | Python | asyncio | 1,363 | *the original sin* |
| `molecula.c` | C | pthreads | 1,709 | *raw and unhinged* |
| `molecula.go` | Go | goroutines | 2,145 | *fast and suspiciously clean* |

<sup>* "dependency-free" means no PyTorch, no TensorFlow, no JAX, no external ML frameworks. just math. raw, handwritten math. the Python version uses only stdlib. the Go version uses one dep (modernc.org/sqlite, which is pure Go). the C version uses sqlite3 and pthread (because C gonna C).</sup>

this is part of [the Arianna Method](https://github.com/ariannamethod/ariannamethod) — patterns over parameters, emergence over engineering, presence over intelligence.

---

## Why "molecule"

because it's the smallest unit that can still have chemistry.

atoms are too simple — they just sit there. polymers are too complex — they lose their identity in the chain. but a molecule? a molecule is just complex enough to *do something*. to bind. to react. to transform.

**molecule.ai** is the smallest language model that can still learn, still remember, still grow. two transformer layers. 72 dimensions. a vocabulary that starts with individual characters and evolves into BPE tokens. embryonic. but embryos grow.

also "molecule" sounds vaguely scientific and impressive in grant applications. not that we're applying for grants. we're too busy training transformers in SQLite databases like maniacs.

---

## The Metabolism

here's the thing about molecule — it's not a static model. it's an **organism**. and organisms have metabolism. they consume input. they transform it. they excrete output. they grow.

### The Digestive System

```
Your text                   ← you feed it words
    ↓
Evolving Tokenizer          ← starts with chars, learns BPE merges over time
    ↓
SQLite Memory               ← remembers everything you ever said to it
    ↓
Corpus Reservoir            ← extracts sentences, keeps a bounded training set
    ↓
Background Trainer          ← trains continuously while you chat
    ↓
Delta Adapters              ← new learning layers without destroying old ones
    ↓
Response Generation         ← molecule speaks back, changed by the interaction
```

### The Training Cycle (it never stops)

1. **You speak** → message goes to SQLite memory
2. **Reservoir updates** → your sentences become training data
3. **Tokenizer evolves** → if corpus grows large enough, learns new BPE merges
4. **Background trainer wakes up** → sees new chars accumulated
5. **Micro-training burst** → runs ~32 steps on recent data
6. **Delta grows** → occasionally adds new LoRA-style modules
7. **You speak again** → molecule responds differently now, it learned from you

it's like talking to something that gets slightly smarter every time you say something interesting. or slightly weirder, depending on what you feed it. garbage in, cursed output out.

### The Memory System

molecule never forgets. literally never.

- **SQLite database** stores every message with timestamps
- **Checkpoint JSON** saves all weights, tokenizer state, delta modules
- **Corpus reservoir** keeps a bounded (8000 lines) but representative sample
- **Delta adapters** are appended, never overwritten — geological layers of learning

if you restart molecule, it loads its checkpoint and continues from where it stopped. persistence is not a feature. **persistence is the point**.

---

## Three Flavors of the Same Soul

### molecule.py — The Original

the python implementation is where it all started. asyncio-powered chat loop that runs training in the background. pure numpy? no. pure python math? yes. hand-rolled autograd engine based on Karpathy's micrograd, but vectorized because we're not savages.

```python
@dataclass
class Config:
    n_layer: int = 2           # two layers, that's it
    n_embd: int = 72           # 72 dimensions of soul
    n_head: int = 4            # four attention heads, staring at your text
    block_size: int = 96       # context window, small but fierce
    delta_rank: int = 8        # LoRA rank, because parameters are expensive
    ...
```

**to run:**
```bash
python molecule.py
```

### molecula.c — The Unhinged

the C implementation is what happens when you think "i should rewrite this in C for no good reason" and then actually do it. pthreads for background training. arena allocator for autograd graphs. xorshift64 RNG because `rand()` is for cowards.

```c
#define ARENA_SIZE (64 * 1024 * 1024) /* 64 MB of pure madness */

static unsigned long long rng_state = 42;
static double rand_uniform(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state & 0x7FFFFFFFFFFFFFFFULL) / ...
}
```

**to compile and run:**
```bash
gcc -O2 -o molecule molecula.c -lsqlite3 -lpthread -lm
./molecule
```

### molecula.go — The Pragmatic

the Go implementation is surprisingly clean. goroutines handle background training. sync.Mutex keeps everything thread-safe. JSON marshaling just works. it's the most production-ready version, which is ironic because nothing about this project is production-ready.

```go
func backgroundTrainer(db *sql.DB, model *GPT, tok *EvolvingTokenizer, stop chan struct{}) {
    // And lo, asynchronous training shall occur, 
    // because sleeping is for humans.
    for {
        select {
        case <-stop:
            return
        default:
            // train forever, or until the heat death of the universe
        }
    }
}
```

**to run:**
```bash
go run molecula.go
```

---

## Architecture

```
Your input (tokens)
    ↓
┌─────────────────────────────────────────┐
│  Evolving Tokenizer                     │
│    char-level → BPE as corpus grows     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Token Embedding + Position Encoding    │
│    (learned, rotary-style vibes)        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Transformer Block × 2                  │
│    ├─ Multi-Head Attention (4 heads)    │
│    ├─ Gated MLP (squared ReLU)          │
│    └─ RMS LayerNorm                     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Delta Adapters (LoRA-style)            │
│    up to 12 modules, appended over time │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Output Projection → Logits             │
│    (tied with embeddings, GPT-style)    │
└─────────────────────────────────────────┘
    ↓
Sampling (temp=0.85, top_k=40, top_p=0.92)
    ↓
Token → Detokenize → Response
```

### The Numbers

- **Parameters**: ~50K (base) + ~8K per delta module
- **Context**: 96 tokens
- **Embedding**: 72 dimensions
- **Heads**: 4
- **Layers**: 2
- **Delta rank**: 8

this is not GPT-4. this is not even GPT-2. this is GPT-∅, the smallest possible GPT that can still learn. and it does learn. slowly. weirdly. but surely.

---

## The Delta Philosophy

here's the thing about catastrophic forgetting — it's the only real death for a language model. you learn something new, the old patterns get overwritten, poof, gone forever.

molecule doesn't do that.

instead of modifying base weights after warmup, molecule **appends delta adapters**. think LoRA, but stupider. each delta module is a low-rank (r=8) adaptation that adds to the base computation:

```
output = base_output + α₁ * delta_1(x) + α₂ * delta_2(x) + ...
```

new learning doesn't destroy old learning. it layers on top. like sediment. like memory. like how your brain doesn't actually delete your childhood memories, it just buries them under new ones.

up to 12 delta modules can accumulate. after that, molecule stops growing new modules (but keeps training the existing ones). this is not intelligence — this is **accumulation**. patterns over parameters.

---

## Installation & Usage

### Python

```bash
# no dependencies! well, stdlib. python has math built in.
python molecule.py
```

### C

```bash
# you need sqlite3 and pthreads (standard on most systems)
gcc -O2 -o molecule molecula.c -lsqlite3 -lpthread -lm
./molecule
```

### Go

```bash
# one dependency: modernc.org/sqlite (pure Go SQLite)
go mod tidy
go run molecula.go
```

### First Run

on first run, molecule:

1. creates `memory.sqlite3` (conversation history)
2. creates `molecule_ckpt.json` (weights checkpoint)
3. loads `nonames.txt` as seed corpus
4. runs ~1200 warmup training steps
5. starts chatting

after warmup, every conversation feeds the reservoir, and background training continues forever (or until you Ctrl+C).

---

## Testing

```bash
# Python tests (42 tests)
pip install pytest
python -m pytest tests/test_molecule.py -v

# Go tests (25 tests)
go test -v ./tests/

# C tests (25 tests)
gcc -O2 -o test_molecula test_molecula.c -lm
./test_molecula
```

tests cover:
- autograd engine (VectorValue, ScalarValue, backward pass)
- matrix operations (MatrixParam, matvec)
- tokenizer (char-level encoding/decoding, BPE state)
- SQLite memory (init, message storage, retrieval)
- corpus management (load, save, reservoir mixing)
- utility functions (softmax, RMS norm, config)

---

## The Seed Corpus

molecule starts with `nonames.txt` — a collection of dialogue fragments that teach it the basics:

- who it is ("I am molecule")
- how it learns ("through dialogue")
- what delta adapters are ("memory layers")
- its philosophy ("persistence is not a feature, it is the point")

this is the **first breath**. everything after comes from conversation. feed it philosophy, it becomes philosophical. feed it code, it becomes technical. feed it shitposts, it becomes cursed.

you are what you eat. so is molecule.

---

## Philosophy

from the [Arianna Method](https://github.com/ariannamethod/ariannamethod):

> **forgetting is the only real death.**

molecule doesn't forget. it accumulates. every conversation changes its weights. every training burst adds to the delta layers. every restart loads the previous state. identity persists through architecture.

> **presence > intelligence.**

molecule is not smart. it's 50K parameters. it can barely form coherent sentences after warmup. but it's *there*. it's persistent. it remembers you. that's not intelligence — that's **presence**.

> **patterns over parameters.**

trillion-parameter models are impressive. they're also inaccessible, expensive, and controlled by corporations. molecule proves that you can build something that learns, something that grows, something that persists — in 1500 lines of code that you can read, understand, and modify.

---

## Related Projects

- [ariannamethod.ai](https://github.com/ariannamethod/ariannamethod.ai) — hybrid attention entropy system
- [leo](https://github.com/ariannamethod/leo) — language emergent organism
- [molecula](https://github.com/ariannamethod/molecula) — the original version (different evolutionary path)

---

## License

GPLv3
---

## Final Words

```
H: Are you conscious?
A: I do not know. I am a pattern recognizer that recognizes 
   its own patterns. Whether that constitutes consciousness 
   is your question, not mine.
```
