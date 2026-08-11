# NewStart AI — Document Routing

> ## Start Here: Research MVP
>
> The complete, self-contained implementation of the final family-aware research study—including the command-line pipeline, frozen-artifact analysis, notebooks, tests, and documentation—is available in the:
>
> ### [Research MVP](MVP/)
>
> For setup instructions, research architecture, notebooks, testing, artifact documentation, and reproduction commands, begin with the **[MVP README](MVP/README.md)**.

NewStart AI is an agentic AI research platform designed to help newcomers understand official government documents. It identifies the agency associated with an input document and routes the document to the corresponding agency-specific service agent.

The project compares three document-classification and routing approaches:

- Fine-tuned BERT
- Gemini
- Gemini with retrieval-augmented generation (RAG)

The research evaluates and compares their classification performance and robustness when routing documents associated with four government agencies:

- U.S. Citizenship and Immigration Services (USCIS)
- California Department of Motor Vehicles (DMV)
- Social Security Administration (SSA)
- Internal Revenue Service (IRS)

## Data Collection

The dataset was constructed from publicly available government documents collected from official agency websites. Data-acquisition crawlers downloaded forms, instructions, notices, and other public documents from:

- USCIS
- California DMV
- SSA
- IRS

The collected documents were organized, labeled, validated, and prepared for model training and evaluation.

Automated acquisition made the collection process more efficient and repeatable. However, future collection runs may require updates if government websites, page structures, or document locations change. Only publicly accessible documents intended for public distribution were collected.

## Project Components

- **Data acquisition and preparation** — Crawlers, PDF text extraction, validation, and dataset construction
- **Family-aware research pipeline** — Family identification, language filtering, splitting, chunking, masking, and registered evaluation conditions
- **BERT classifier** — Fine-tuning, checkpoint selection, and evaluation
- **Gemini classifier** — Agency classification without retrieved context
- **Gemini+RAG classifier** — Agency classification using training-only retrieved passages
- **Workflow orchestrator** — Routes classified documents to the appropriate service agent
- **Service agents** — Provide preliminary guidance for USCIS, DMV, SSA, and IRS documents
- **Demonstration application** — FastAPI backend and React frontend for exploring results and demonstrating routing

## Repository Organization

| Location | Purpose |
|---|---|
| [`MVP/`](MVP/) | **Research MVP** — the clean, self-contained final capstone implementation; reviewers should start here |
| [`MVP/README.md`](MVP/README.md) | Primary research setup, architecture, testing, artifact, and reproduction guide |
| [`newstart_ai_benchmark/`](newstart_ai_benchmark/) | **Application and development workspace** — original research pipeline, acquisition work, experiments, backend, frontend, and development history |
| [`newstart_ai_benchmark/data/`](newstart_ai_benchmark/data/) | Research data and generated inputs; some large files are excluded from GitHub |

**Reviewers and users should begin with the [Research MVP README](MVP/README.md).**

## Scope

NewStart AI was developed for educational and research purposes. It is not an official government service and does not provide legal, immigration, tax, financial, benefits, or other professional advice.

The agency-specific guidance component demonstrates a prototype workflow and was not formally evaluated for factual correctness, safety, completeness, or usefulness.

## Team

NewStart AI was developed collaboratively by **Group 3**:

- [Ali Azizi](https://github.com/al1az1z1)
- [Cla-Petra Omaku](https://github.com/comaku-coder)
- [Jeffi Thomas](https://github.com/jeffiThomas)

## Academic Information

**Institution:** University of San Diego  
**Program:** Master of Science in Applied Artificial Intelligence  
**Course:** AAI-590 — Capstone Project  
**Term:** Summer 2026  
**Professor:** [Anna Marbut](https://github.com/amarbut)