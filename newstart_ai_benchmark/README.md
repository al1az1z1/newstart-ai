# NewStart AI — Application and Development Workspace

> **Reviewing the final capstone research?**
> Begin with the root-level **[Research MVP](../MVP/README.md)**. It provides the cleaned research implementation, final notebooks, frozen-results analysis, testing instructions, artifact documentation, and safe reproduction commands.

This directory, `newstart_ai_benchmark/`, is the broader NewStart AI application, research-development, and experimental workspace. It contains the original research pipeline, data-acquisition work, family-aware experiments, frozen artifacts, FastAPI backend, React frontend, and project development history from which the Research MVP was prepared.

The two directories provide complementary views of the same project:

| Project area                         | Primary purpose                                                                                                  |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| **[Research MVP](../MVP/README.md)** | Stable and self-contained implementation corresponding to the submitted capstone report and presentation         |
| **`newstart_ai_benchmark/`**         | Broader application, original research pipeline, experiments, data acquisition, artifacts, backend, and frontend |

## Project Overview

NewStart AI investigates intelligent routing of United States government documents to four agencies:

* U.S. Citizenship and Immigration Services (USCIS)
* California Department of Motor Vehicles (DMV)
* Social Security Administration (SSA)
* Internal Revenue Service (IRS)

The capstone study compares three document-classification approaches:

1. Fine-tuned BERT
2. Gemini
3. Gemini with retrieval-augmented generation (RAG)

The primary research evaluation uses a family-aware dataset split designed to prevent related forms, instructions, supplements, and translations from appearing across training, validation, and test partitions.

The three approaches were evaluated using the same frozen set of 99 family-disjoint test documents under ten masked and unmasked conditions, producing 2,970 evaluation cases.

For the final methodology, results, research notebooks, and reproduction guidance, see the **[Research MVP README](../MVP/README.md)**.

## Workspace Components

| Location                               | Purpose                                                                                                            |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| [`src/newstart_ai/`](src/newstart_ai/) | Original research pipeline for data preparation, family-aware splitting, classification, retrieval, and evaluation |
| [`backend/`](backend/)                 | FastAPI application providing research-results, replay, demo, and live-routing endpoints                           |
| [`frontend/`](frontend/)               | Vite and React interface for exploring results and demonstrating the NewStart AI workflow                          |
| [`configs/`](configs/)                 | Configuration files for the base pipeline, BERT, family-aware experiments, Gemini, and RAG                         |
| [`data/`](data/)                       | Raw, processed, split, family-aware, chunked, masked, and condition-registered research data                       |
| [`artifacts/`](artifacts/)             | Models, predictions, manifests, reports, caches, embeddings, and vector stores                                     |
| [`prompts/`](prompts/)                 | Classification, RAG-classification, and agency-guidance prompt templates                                           |
| [`notebooks/`](notebooks/)             | Data acquisition, historical experiments, and family-aware analyses                                                |
| [`tests/`](tests/)                     | Tests covering the original research pipeline and experimental stages                                              |
| [`backend/tests/`](backend/tests/)     | API endpoint and application-service tests                                                                         |
| [`requirements/`](requirements/)       | Base, development, and notebook dependency specifications                                                          |

## Repository Structure

```text
newstart_ai_benchmark/
├── README.md
├── pyproject.toml
│
├── src/newstart_ai/
│   ├── config/
│   ├── data/
│   ├── models/
│   │   ├── bert/
│   │   └── llm/
│   ├── rag/
│   ├── agents/
│   ├── routing/
│   ├── eda/
│   ├── evaluation/
│   ├── schemas/
│   └── common/
│
├── backend/
│   ├── app/
│   └── tests/
│
├── frontend/
│   └── src/
│
├── configs/
├── data/
├── artifacts/
│   └── family_aware/
├── prompts/
├── notebooks/
│   ├── 00_data_acquisition/
│   ├── 01-10
│   └── 11-16
├── tests/
├── requirements/
├── scripts/
└── docs/
```

## Research Pipeline

The original pipeline under `src/newstart_ai/` contains the implementation used to develop and execute the project’s experiments, including:

* Dataset validation and label auditing
* Language filtering and manual corrections
* Document-family identification
* Family-aware train, validation, and test splitting
* Token-aware chunking
* Agency-identifier masking
* Generation of ten registered evaluation conditions
* Family-aware BERT training and checkpoint selection
* Gemini classification
* Gemini+RAG classification
* ChromaDB index construction and retrieval
* Condition-level evaluation
* Model comparison and error analysis
* Agency-specific routing and guidance logic

This codebase preserves the original implementation and experimental development history. The root-level Research MVP contains a simplified and reorganized copy of the required research logic and does not import executable code from this workspace at runtime.

## Application

The workspace includes a complete demonstration application.

### Backend

The [`backend/`](backend/) directory contains a FastAPI application with endpoints and services for:

* Displaying family-aware research results
* Replaying frozen predictions
* Comparing results across evaluation conditions
* Demonstrating document classification and agency routing
* Supporting live prototype workflows where configured

### Frontend

The [`frontend/`](frontend/) directory contains the Vite and React user interface, including:

* Research-results pages
* Whole-test evaluation summaries
* Random-document prediction replay
* Condition selection
* Interactive document-routing demonstrations
* Supporting components, hooks, routes, services, and utilities

The application demonstrates the feasibility of document classification and agency routing. The downstream guidance component was not formally evaluated for factual correctness, safety, completeness, or usefulness.

## Research Notebooks

The notebook collection preserves the development of the project:

| Notebook group         | Purpose                                                                   |
| ---------------------- | ------------------------------------------------------------------------- |
| `00_data_acquisition/` | Government-document acquisition and initial dataset construction          |
| `01-10`                | Original non-family-aware pipeline and historical experiments             |
| `11-16`                | Family-aware EDA, training, evaluation, comparison, and error attribution |

The cleaned research equivalents of the final family-aware notebooks are available under [`../MVP/notebooks/`](../MVP/notebooks/).

## Configuration

The primary configuration files are located in [`configs/`](configs/):

* `base.yaml`
* `bert.yaml`
* `family_aware.yaml`
* `llm.yaml`
* `rag.yaml`

The Research MVP automatically treats this directory as its sibling benchmark workspace. If the directories are stored separately, the MVP supports an alternative location through:

```text
NEWSTART_BENCHMARK_ROOT
```

Configuration and usage instructions are provided in the **[MVP README](../MVP/README.md)**.

## Artifact Availability

The workspace contains or references research artifacts under:

```text
artifacts/
artifacts/family_aware/
data/
data/family_aware_splits/
```

Some generated files are too large for normal GitHub distribution and may be excluded through `.gitignore`, including:

* BERT checkpoint weights
* Gemini embedding caches
* Chroma vector stores
* API caches
* Large document collections
* Temporary experiment outputs

Therefore, a fresh clone may not contain every artifact required to rerun training, hosted-model inference, embedding generation, or retrieval-index construction.

Smaller research records—such as configurations, manifests, metrics, evaluation tables, and frozen predictions—are retained where practical. The submitted frozen results should be treated as the official experimental record because hosted-model outputs may change if rerun later.

For detailed artifact provenance and regeneration guidance, see [`../MVP/docs/ARTIFACTS.md`](../MVP/docs/ARTIFACTS.md).

## Development Setup

Install the benchmark package from this directory:

```bash
python -m pip install -e .
```

Development and notebook dependencies are listed under [`requirements/`](requirements/).

Before running training, Gemini evaluation, embedding generation, or Chroma index construction, review the relevant configuration files and the safe-rerun instructions in [`../MVP/docs/RERUN_GUIDE.md`](../MVP/docs/RERUN_GUIDE.md).

These operations may require:

* Local research datasets
* Model checkpoints
* API credentials
* Additional Python dependencies
* Considerable computation time
* Writable artifact directories

## Testing

Run the original research-pipeline tests from `newstart_ai_benchmark/`:

```bash
pytest tests/
```

Run the backend tests separately:

```bash
pytest backend/tests/
```

The cleaned Research MVP has an independent test suite under [`../MVP/tests/`](../MVP/tests/). Its tests focus on safe default behavior, artifact integrity, result recomputation, path resolution, and independence from the original `newstart_ai` package.

## Research and Application Boundaries

The repository separates the project into two clear areas:

### Research MVP

Use [`../MVP/`](../MVP/) when you want to:

* Review the submitted capstone research
* Examine the final family-aware methodology
* Inspect frozen experimental results
* Run safe comparison commands
* Execute the seven cleaned research notebooks
* Verify the submitted metrics
* Understand artifact provenance and rerun requirements

### Application and Development Workspace

Use this directory when you want to:

* Inspect the original research implementation
* Review the complete experiment history
* Work with the FastAPI backend
* Work with the React frontend
* Examine data-acquisition code
* Explore original and family-aware notebooks
* Inspect prompts, configurations, artifacts, and tests
* Continue application or experimental development

## Scope and Limitations

NewStart AI is a research prototype and demonstration platform. It is not an official government service and does not provide legal, immigration, tax, financial, benefits, or other professional advice.

Important limitations include:

* The study covers four government agencies.
* The IRS class is considerably smaller than the other classes.
* The final family-aware test set contains 99 eligible documents.
* Some large artifacts are excluded from GitHub.
* Gemini outputs may vary across future reruns.
* RAG performance depends on the available corpus, embeddings, and retrieval configuration.
* The agency-guidance component was not formally evaluated for correctness or safety.
* Strong results on the frozen research dataset should not be interpreted as universal performance on arbitrary government documents.

## Authors

NewStart AI was developed collaboratively by **Group 3**:

* **Ali Azizi**
* **Cla-Petra Omaku**
* **Gaius Thomas**

**Institution:** University of San Diego
**Program:** Master of Science in Applied Artificial Intelligence
**Course:** AAI-590 — Capstone Project
**Term:** Summer 2026

Detailed academic information, contributions, and the AI-assisted development disclosure are available in the **[Research MVP README](../MVP/README.md)**.

## Repository

The complete project repository is available at:

[github.com/al1az1z1/newstart-ai](https://github.com/al1az1z1/newstart-ai)

For the completed capstone study, start with the **[Research MVP](../MVP/README.md)**. Use this workspace for the broader application, original research implementation, experiment history, data acquisition, and development context.
