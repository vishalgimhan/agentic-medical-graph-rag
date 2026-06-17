# Agentic Medical Graph RAG

An agentic **GraphRAG** (Retrieval-Augmented Generation over a knowledge graph) system for answering
drug-interaction and medical questions. It combines a **Neo4j** knowledge graph of drugs, conditions,
side effects, and drug classes with a **LangGraph**-orchestrated CRAG (Corrective RAG) agent that
routes, retrieves, grades, rewrites, and generates grounded, citation-backed answers.

> ⚕️ **Disclaimer:** This project is for educational and research purposes only. It is **not** medical
> advice. Always consult a qualified healthcare professional for medical decisions.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Knowledge Graph Schema](#knowledge-graph-schema)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [The Agent Pipeline](#the-agent-pipeline)
- [Notebooks](#notebooks)
- [Dataset](#dataset)
- [Development Notes](#development-notes)

---

## Features

- **Knowledge graph backend** — Drugs, conditions, side effects, and drug classes modeled as a
  property graph in Neo4j with uniqueness constraints and lookup indexes.
- **Multi-hop graph retrieval** — Answers questions that require traversing relationships
  (e.g. "Which drugs in the same class as Warfarin interact with NSAIDs?").
- **Agentic CRAG workflow** — A LangGraph state machine that *routes* questions, *retrieves* from the
  graph, *grades* relevance, *rewrites* failed queries, and *generates* grounded answers.
- **Entity extraction & resolution** — LLM-based extraction of medical entities from natural-language
  questions, resolved against canonical graph nodes (with alias support).
- **Provider-agnostic LLM layer** — Swap between OpenRouter, OpenAI, Anthropic, Google, Groq, and
  DeepSeek via YAML config without touching code. A two-model architecture uses a fast/cheap model for
  routing and generation and a strong model for extraction.
- **Reproducible ETL** — Scripts to initialize the graph schema and seed it from a structured dataset.
- **Teaching notebooks** — Cypher fundamentals, knowledge-graph ETL, and the full agentic GraphRAG build.

---

## Architecture

```
                       ┌─────────────────────────────────────────────┐
   User question  ───► │              LangGraph CRAG Agent            │
                       │                                              │
                       │   route ─► retrieve ─► grade ─► generate     │
                       │                 ▲         │                  │
                       │                 └─ rewrite◄┘ (on low relevance)
                       └───────┬──────────────────────┬───────────────┘
                               │                      │
                     ┌─────────▼────────┐   ┌─────────▼─────────┐
                     │  LLM Services    │   │   Neo4j Client    │
                     │ (general/strong/ │   │ (graph retrieval, │
                     │   embeddings)    │   │  multi-hop Cypher)│
                     └─────────┬────────┘   └─────────┬─────────┘
                               │                      │
                     ┌─────────▼────────┐   ┌─────────▼─────────┐
                     │ LLM Provider     │   │  Neo4j Database   │
                     │ (OpenRouter/...) │   │  (knowledge graph)│
                     └──────────────────┘   └───────────────────┘
```

The system has two phases:

1. **Build phase (ETL)** — Load the structured drug dataset, create the graph schema, and seed nodes
   and relationships into Neo4j (`scripts/init_graph.py`, `scripts/seed_graph.py`).
2. **Query phase (agent)** — At question time, the LangGraph agent extracts entities, retrieves graph
   context, grades it, optionally rewrites the query, and generates a grounded answer.

---

## Knowledge Graph Schema

**Node labels**

| Label        | Key (unique) | Description                                  |
|--------------|--------------|----------------------------------------------|
| `Drug`       | `name`       | A medication (with `aliases`, `description`) |
| `Condition`  | `name`       | A medical condition / indication             |
| `SideEffect` | `name`       | An adverse effect                            |
| `DrugClass`  | `name`       | A pharmacological class                       |

**Relationships**

| Relationship          | Pattern                                      | Notes                                                   |
|-----------------------|----------------------------------------------|---------------------------------------------------------|
| `INTERACTS_WITH`      | `(Drug)-[:INTERACTS_WITH]->(Drug)`           | Has `severity`, `mechanism`, `effect`, `recommendation` |
| `TREATS`              | `(Drug)-[:TREATS]->(Condition)`              |                                                         |
| `CAUSES_SIDE_EFFECT`  | `(Drug)-[:CAUSES_SIDE_EFFECT]->(SideEffect)` |                                                         |
| `CONTRAINDICATED_FOR` | `(Drug)-[:CONTRAINDICATED_FOR]->(Condition)` |                                                         |
| `BELONGS_TO_CLASS`    | `(Drug)-[:BELONGS_TO_CLASS]->(DrugClass)`    |                                                         |

Constraints enforce uniqueness on every node `name`; matching indexes accelerate lookups.

---

## Project Structure

```
agentic-medical-graph-rag/
├── config/
│   ├── models.yaml          # Model names per provider and tier (chat/embedding)
│   └── params.yaml          # Provider, LLM, embedding, retrieval, agent settings
├── data/
│   └── raw/
│       └── drug_interactions.json   # Source dataset (drugs + interactions + passages)
├── notebooks/
│   ├── cypher_fundamentals.ipynb     # Cypher query language tutorial
│   ├── knowledge_graph_and_etl.ipynb # Schema design + ETL walkthrough
│   └── agentic_graph_rag.ipynb       # Full LangGraph CRAG agent build
├── scripts/
│   ├── init_graph.py        # Create constraints/indexes (optional --clean)
│   └── seed_graph.py        # Load dataset into Neo4j (optional --clean-first)
├── utils/
│   ├── config.py            # Central config loader (YAML + env vars)
│   ├── llm_services.py      # LLM + embedding factories (provider-agnostic)
│   ├── neo4j_client.py      # Thread-safe Neo4j driver wrapper
│   └── prompts.py           # All agent prompts (router, grader, generator, ...)
├── main.py                  # Entry-point stub
├── Makefile                 # install / init-graph / seed-graph / clean-graph
├── pyproject.toml           # Project metadata + dependencies (uv)
├── requirements.txt         # Pinned dependencies (pip)
├── .env.example             # Template for required secrets
└── .python-version          # Python 3.12
```

---

## Tech Stack

- **Python 3.12**
- **Neo4j 6.x** — graph database (Aura cloud or local)
- **LangGraph 1.x** — agent orchestration (state machine)
- **LangChain (OpenAI bindings)** — LLM + embedding clients
- **Pydantic 2** — dataset and schema validation
- **PyYAML** — configuration
- **loguru** — logging
- **uv** / **pip** — dependency management

---

## Prerequisites

1. **Python 3.12** (see `.python-version`).
2. **A Neo4j database** — either:
   - [Neo4j Aura](https://neo4j.com/cloud/aura/) (managed, free tier available), or
   - A local Neo4j instance / Docker container.
3. **An LLM API key** — by default an [OpenRouter](https://openrouter.ai/) key. Other providers
   (OpenAI, Anthropic, Google, Groq, DeepSeek) are supported via config.

---

## Setup

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd agentic-medical-graph-rag

# 2. Install dependencies
#    Option A — pip:
pip install -r requirements.txt
#    (equivalently: make install)
#
#    Option B — uv (uses pyproject.toml + uv.lock):
uv sync

# 3. Configure secrets
cp .env.example .env
#    then edit .env with your Neo4j and LLM credentials

# 4. Initialize the graph schema (constraints + indexes)
python scripts/init_graph.py          # or: make init-graph

# 5. Seed the graph with the drug interaction dataset
python scripts/seed_graph.py          # or: make seed-graph
```

---

## Configuration

Configuration is split between **secrets** (`.env`) and **non-secret settings** (`config/*.yaml`).

### `.env` (secrets)

```dotenv
PROJECT_NAME = agentic_medical_graph_rag

NEO4J_URI=<your_neo4j_uri>            # e.g. neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USERNAME=<your_neo4j_username>  # e.g. neo4j
NEO4J_PASSWORD=<your_neo4j_password>
NEO4J_DATABASE=<your_neo4j_database>  # e.g. neo4j

OPENROUTER_API_KEY=<your_openrouter_api_key>
```

> Provider key names are auto-mapped in `utils/config.py` (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
> `GOOGLE_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, ...). Add whichever your active provider needs.

### `config/params.yaml` (behavior)

Key sections:

- **`provider`** — `default` provider (`openrouter`), model `tier`, and the OpenRouter base URL.
- **`llm`** — `temperature` (0.0), `max_tokens` (2000), `streaming`.
- **`embedding`** — `tier` (`small` = 1536d / `default` = 3072d), `batch_size`, `show_progress`.
- **`extraction`** — ETL chunking and the model tier (`strong`) used for entity extraction.
- **`retrieval`** — `max_hops` (2), `top_k` (5), similarity / fuzzy-match thresholds.
- **`agent`** — `max_retries` for query rewrites, model tier for agent nodes.
- **`paths`** — data directories.

### `config/models.yaml` (model names)

Maps each provider to concrete chat and embedding model names by tier
(`general`, `strong`, `reason`). The **two-model architecture**:

- **General model** (`gpt-4o-mini` via OpenRouter) — routing, grading, generation.
- **Strong model** (`gpt-4o` via OpenRouter) — entity extraction & resolution.

To switch providers, change `provider.default` in `params.yaml` and ensure the matching API key is in
`.env`. To switch models, edit `models.yaml`.

---

## Usage

### Make targets

```bash
make install      # Install dependencies from requirements.txt
make init-graph   # Create Neo4j constraints and indexes
make seed-graph   # Populate Neo4j with the drug interaction dataset
make clean-graph  # Delete all graph data (init_graph.py --clean)
```

### Scripts directly

```bash
# Re-initialize, wiping existing data first
python scripts/init_graph.py --clean

# Seed after cleaning
python scripts/seed_graph.py --clean-first
```

Both scripts verify connectivity and print a summary of the resulting graph
(node/relationship counts by label).

### Using the utilities in code

```python
from dotenv import load_dotenv
load_dotenv()

from utils import (
    get_general_llm, get_strong_llm, get_embeddings,
    get_neo4j_client, validate, dump,
)

validate()   # check secrets + create data dirs
dump()       # print active (non-secret) configuration

# Query the graph
with get_neo4j_client() as client:
    client.verify_connectivity()
    rows = client.query(
        "MATCH (a:Drug {name:$n})-[r:INTERACTS_WITH]-(b:Drug) "
        "RETURN b.name AS drug, r.severity AS severity",
        {"n": "Warfarin"},
    )
    print(rows)

# Call an LLM
llm = get_general_llm()
print(llm.invoke("List two NSAIDs.").content)
```

The end-to-end agent is built and demonstrated in
[`notebooks/agentic_graph_rag.ipynb`](notebooks/agentic_graph_rag.ipynb).

---

## The Agent Pipeline

The CRAG (Corrective RAG) agent is a LangGraph state machine. Prompts for each node live in
[`utils/prompts.py`](utils/prompts.py):

| Stage              | Prompt                                         | Role |
|--------------------|------------------------------------------------|------|
| **Route**          | `ROUTER_SYSTEM_PROMPT`                          | Classify question → `drug_interaction` / `general_medical` / `chitchat`. |
| **Extract**        | `ENTITY_EXTRACTION_PROMPT`                      | Pull drug/condition/side-effect/class entities from the question. |
| **Retrieve**       | (graph traversal)                              | Resolve entities to nodes; run multi-hop Cypher to gather context. |
| **Grade**          | `GRADER_SYSTEM_PROMPT` / `..._HUMAN`           | Judge whether retrieved context is relevant (`yes`/`no`). |
| **Rewrite**        | `REWRITER_SYSTEM_PROMPT` / `..._HUMAN`         | On failure, reformulate the query (generics, expand abbreviations) and retry. |
| **Generate**       | `GENERATOR_SYSTEM_PROMPT` / `..._HUMAN`        | Produce a grounded answer citing graph relationships and severities. |
| **General / Chat** | `GENERAL_MEDICAL_PROMPT` / `CHITCHAT_RESPONSE` | Handle non-graph questions and small talk. |

Conditional edges loop **grade → rewrite → retrieve** up to `agent.max_retries` times before
generating from whatever context is available. The generator is instructed to use only the provided
context, cite relationships and severities, and always append a "consult a professional" disclaimer.

There is also an **ETL extraction** prompt pair (`EXTRACTION_SYSTEM_PROMPT` /
`EXTRACTION_HUMAN_PROMPT`) for extracting entities and relationships from free-text passages using the
strong model.

---

## Notebooks

Run in order for a guided tour:

1. **`cypher_fundamentals.ipynb`** — Cypher query language: node/relationship patterns, `MATCH` vs
   `OPTIONAL MATCH`, filtering, projection, aggregations.
2. **`knowledge_graph_and_etl.ipynb`** — Graph schema design, constraints/indexes, and loading data
   into Neo4j.
3. **`agentic_graph_rag.ipynb`** — Building the full agent: state, prompts/schemas, entity resolution,
   the graph retriever, multi-hop queries, the CRAG nodes, LangGraph assembly, and tests.

---

## Dataset

[`data/raw/drug_interactions.json`](data/raw/drug_interactions.json) is a curated, structured dataset:

- **80 drugs** across **23 drug classes** (NSAID, Anticoagulant, Statin, SSRI, Beta-Blocker, ACE
  Inhibitor, Opioid, Benzodiazepine, and more).
- **244 drug-drug interactions**, each with `severity` (Major / Moderate / Minor), `mechanism`,
  `effect`, and `recommendation`.
- Per-drug `aliases`, `description`, `treats`, `side_effects`, and `contraindications`.
- A set of free-text `text_passages` available for LLM-based extraction experiments.

`scripts/seed_graph.py` validates this file with Pydantic models (`DrugRecord`, `InteractionRecord`,
`DrugInteractionDataset`) before batched `UNWIND` writes into the graph.

---

## Development Notes

- **Config is centralized** in `utils/config.py`; import constants/helpers from `utils` rather than
  reading env/YAML elsewhere. Call `validate()` early to fail fast on missing secrets.
- **Neo4j access** goes through `Neo4jClient`, a context-managed, thread-safe wrapper offering
  `write`, `batch_write` (UNWIND), `query`, and graph-introspection helpers.
- **Embedding dimensions** are inferred from the model name (1536 for `*-small`/`ada`, 3072 for
  `*-large`).
- **Provider portability** — the LLM factory builds `ChatOpenAI`/`OpenAIEmbeddings` clients and only
  swaps the base URL + key per provider, so OpenAI-compatible providers work out of the box.
- `.env`, `.venv`, and `*.lock` files are git-ignored.
