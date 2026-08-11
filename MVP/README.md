# NewStart AI — Research MVP

**Newcomer Navigator: Agentic AI for Document Understanding and Guidance**

NewStart AI is a research-oriented document-routing system developed to compare three approaches for classifying United States government documents:

- Fine-tuned BERT
- Gemini
- Gemini with retrieval-augmented generation (RAG)

The system routes documents to one of four agencies:

- U.S. Citizenship and Immigration Services (USCIS)
- California Department of Motor Vehicles (DMV)
- Social Security Administration (SSA)
- Internal Revenue Service (IRS)

This root-level `MVP/` directory is the recommended starting point for professors, reviewers, researchers, and users interested in the completed capstone study. It corresponds to the submitted research report, presentation, methodology, and frozen experimental results.

The sibling [`newstart_ai_benchmark/`](../newstart_ai_benchmark/) directory contains the broader application and development workspace, including the original experiments, data-acquisition work, application components, datasets, and research artifacts.

## Research Overview

Government forms and instructions often share identifiers, templates, vocabulary, and closely related versions. A conventional random document split can therefore place members of the same document family in both training and testing data, potentially producing overly optimistic performance estimates.

To reduce this risk, the primary study used a family-aware split that kept related documents together within a single partition. The study then evaluated model performance under ten registered input conditions representing complete documents, selected document regions, and masked versions with identifying information removed.

The three methods were evaluated on the same frozen, family-disjoint test set:

- 99 test documents
- 10 masked and unmasked conditions
- 3 classification methods
- 2,970 total evaluation cases

Macro F1 was used as the primary evaluation metric because the dataset was substantially imbalanced across the four agencies.

## Headline Result

The primary study used a frozen, family-disjoint test set to reduce information leakage between related government documents. Each of the three classification methods was evaluated using the same 99 test documents under ten registered input conditions:

1. `complete_unmasked`
2. `beginning_only_unmasked`
3. `middle_only_unmasked`
4. `end_only_unmasked`
5. `beginning_middle_end_unmasked`
6. `complete_masked`
7. `beginning_only_masked`
8. `middle_only_masked`
9. `end_only_masked`
10. `beginning_middle_end_masked`

These conditions evaluated model robustness when only selected document regions were available or when explicit identifiers—including agency names, form numbers, OMB control numbers, and government URLs—were masked.

The same registered input text was provided to Fine-tuned BERT, Gemini, and Gemini+RAG for every document-condition pair:

- 99 test documents
- 10 evaluation conditions
- 3 classification methods
- 2,970 total evaluation cases

Under the primary `complete_unmasked` condition, all three methods achieved an accuracy and Macro F1 score of `1.000`. Complete condition-level results are available in the **[Research MVP](MVP/README.md)**.

The remaining nine conditions evaluate robustness when only certain document regions are available or when explicit agency identifiers, form numbers, OMB control numbers, and government URLs are masked.

Detailed condition-level results, confusion matrices, reliability comparisons, latency measurements, and error analysis are available in the notebooks, especially:

- [`07_final_comparison.ipynb`](notebooks/07_final_comparison.ipynb)
- [`06_bert_error_attribution.ipynb`](notebooks/06_bert_error_attribution.ipynb)

## Research Design

The final family-aware research pipeline includes:

1. Dataset validation and label auditing
2. Language filtering and manual corrections
3. Document-family identification
4. Family-aware train, validation, and test splitting
5. Token-aware document chunking
6. Identifier masking
7. Generation of ten registered evaluation conditions
8. BERT fine-tuning and checkpoint selection
9. Gemini classification
10. Gemini+RAG classification
11. Frozen test evaluation and final comparison

### Family-aware split

Related forms, instructions, supplements, and translations were grouped into document families using form numbers and filename-derived identifiers. All members of a family were assigned to the same partition.

The final eligible dataset was split into:

| Partition | Documents | Families |
|---|---:|---:|
| Training | 461 | 355 |
| Validation | 99 | 78 |
| Test | 99 | 77 |

The final split contained zero document overlap and zero family overlap between partitions.

### Evaluation conditions

Each test document was evaluated under ten conditions:

1. `complete_unmasked`
2. `beginning_only_unmasked`
3. `middle_only_unmasked`
4. `end_only_unmasked`
5. `beginning_middle_end_unmasked`
6. `complete_masked`
7. `beginning_only_masked`
8. `middle_only_masked`
9. `end_only_masked`
10. `beginning_middle_end_masked`

The same registered text was supplied to all three methods for each document-condition pair.

### BERT configuration

The family-aware BERT model used:

| Setting | Value |
|---|---|
| Base model | `bert-base-uncased` |
| Maximum epochs | 6 |
| Batch size | 8 |
| Optimizer | AdamW |
| Learning rate | $2 \times 10^{-5}$ |
| Loss | Weighted cross-entropy |
| Maximum window size | 512 tokens |
| Window overlap | 128 tokens |
| Selection metric | Validation Macro F1 |
| Tie rule | Earliest epoch achieving the maximum score |

Class weighting addressed agency imbalance, while inverse chunk-count weighting prevented long documents from dominating the training objective. Epoch 2 was retained as the earliest checkpoint achieving the maximum validation Macro F1.

### Gemini configuration

Plain Gemini used:

- Temperature 0
- A structured JSON response schema
- Four permitted agency labels
- A 6,000-character input limit
- Recorded truncation status
- No document-window aggregation
- No retrieved context

### Gemini with RAG

Gemini+RAG used the same classification setup with additional passages retrieved from the training partition only.

The RAG pipeline used:

- `gemini-embedding-001`
- ChromaDB vector stores
- Separate masked and unmasked indexes
- Top-$k=3$ retrieval
- Approximately 8,600 indexed training chunks
- No test documents in the retrieval indexes

## MVP Architecture

`MVP/` is a self-contained Python package. The implementation under `newstart_ai_mvp/` contains its own simplified and reorganized copy of the required research pipeline logic.

It does not import executable Python code from:

```text
newstart_ai_benchmark/src/newstart_ai/
```

The original source remains available as reference and development history. The MVP may read documented datasets and frozen research artifacts from the sibling `newstart_ai_benchmark/` workspace, but it has no runtime dependency on the original `newstart_ai` package.

The implementation supports three execution categories:

- Safe inspection of frozen research artifacts
- Offline recomputation and verification
- Explicitly enabled expensive reruns

See [`docs/STAGES.md`](docs/STAGES.md) for the complete stage model.

## Quickstart

### Installation

From the repository root:

```bash
cd MVP
python -m pip install -e .
```

### Display the command menu

```bash
python -m newstart_ai_mvp
```

### Display the headline comparison

```bash
python -m newstart_ai_mvp.compare_models
```

Default commands are safe and read-only. They inspect existing frozen artifacts without training a model, calling Gemini, generating embeddings, rebuilding a Chroma index, or overwriting submitted results.

Training and external-service operations require an explicit flag such as:

```text
--run
--run-training
--run-api
--rebuild-index
```

Review [`docs/RERUN_GUIDE.md`](docs/RERUN_GUIDE.md) before executing an expensive stage.

## Repository Layout

The repository separates the cleaned capstone research implementation from the broader application and development workspace:

```text
newstart-ai/
├── README.md
├── newstart_ai_benchmark/              # Application and development workspace
└── MVP/
    ├── README.md                       # Primary MVP entry point
    ├── pyproject.toml                  # Package and dependency configuration
    │
    ├── newstart_ai_mvp/                # Self-contained research package
    │   ├── __init__.py
    │   ├── __main__.py                 # Command menu
    │   ├── config.py                   # Workspace path resolution
    │   ├── run_scope.py                # Safe output isolation
    │   ├── cli_common.py
    │   ├── artifact_report.py          # Read-only artifact summaries
    │   ├── data_pipeline.py
    │   ├── bert_pipeline.py
    │   ├── llm_pipeline.py
    │   ├── rag_pipeline.py
    │   ├── stage1_validate_and_audit.py
    │   ├── stage2_language_filter.py
    │   ├── stage3_family_split.py
    │   ├── stage4_chunk_and_mask.py
    │   ├── stage5_build_conditions.py
    │   ├── prepare_data.py
    │   ├── train_bert.py
    │   ├── evaluate_bert.py
    │   ├── build_rag_index.py
    │   ├── evaluate_llm.py
    │   ├── evaluate_rag.py
    │   └── compare_models.py
    │
    ├── notebooks/
    │   ├── 00_data_acquisition/        # Six original acquisition notebooks
    │   ├── 01_frozen_split_and_chunking.ipynb
    │   ├── 02_bert_training_summary.ipynb
    │   ├── 03_bert_test_evaluation.ipynb
    │   ├── 04_llm_evaluation_summary.ipynb
    │   ├── 05_llm_rag_evaluation_summary.ipynb
    │   ├── 06_bert_error_attribution.ipynb
    │   └── 07_final_comparison.ipynb
    │
    ├── docs/
    │   ├── ARTIFACTS.md                # Artifact provenance and regeneration
    │   ├── STAGES.md                   # Pipeline execution model
    │   └── RERUN_GUIDE.md              # Safe rerun instructions
    │
    ├── tests/                          # 85 automated tests
    │   ├── conftest.py
    │   ├── test_benchmark_root_resolution.py
    │   ├── test_no_original_source_dependency.py
    │   ├── test_default_mode_is_read_only.py
    │   ├── test_no_network_by_default.py
    │   ├── test_artifact_schemas.py
    │   ├── test_recomputed_metrics_match_report.py
    │   ├── test_run_scope_never_touches_frozen.py
    │   └── test_cli_help_lists_every_stage.py
    │
    ├── runs/                           # Isolated outputs from future reruns
    └── app/demo/                       # Demonstration PDF documents
```

The `runs/` directory remains empty, except for its placeholder file, until an explicitly enabled rerun is performed. New outputs are isolated there so that submitted experimental results are not overwritten.

## Research Notebooks

| Notebook | Purpose |
|---|---|
| `01_frozen_split_and_chunking.ipynb` | Family-aware splitting, frozen partitions, and document chunking |
| `02_bert_training_summary.ipynb` | BERT training progression and checkpoint selection |
| `03_bert_test_evaluation.ipynb` | BERT evaluation across the registered conditions |
| `04_llm_evaluation_summary.ipynb` | Plain Gemini evaluation |
| `05_llm_rag_evaluation_summary.ipynb` | Gemini+RAG evaluation and retrieval analysis |
| `06_bert_error_attribution.ipynb` | BERT error analysis and Integrated Gradients attribution |
| `07_final_comparison.ipynb` | Final comparison of all three classification methods |

The notebooks under `notebooks/00_data_acquisition/` preserve the original acquisition and exploratory work as supporting project context.

## Configuration

By default, the MVP expects this sibling-directory structure:

```text
newstart-ai/
├── MVP/
└── newstart_ai_benchmark/
```

The default benchmark root is `repository-root/newstart_ai_benchmark/`. It provides access to documented configuration, data, and artifact paths used by the MVP.

To use a benchmark workspace in another location, set `NEWSTART_BENCHMARK_ROOT` before running an MVP command.

Linux or macOS:

```bash
export NEWSTART_BENCHMARK_ROOT=/path/to/newstart_ai_benchmark
```

Windows PowerShell:

```powershell
$env:NEWSTART_BENCHMARK_ROOT = "D:\path\to\newstart_ai_benchmark"
```

Windows Command Prompt:

```cmd
set NEWSTART_BENCHMARK_ROOT=D:\path\to\newstart_ai_benchmark
```

See [`newstart_ai_mvp/config.py`](newstart_ai_mvp/config.py) for the path-resolution implementation.

## Artifact Availability

Several large generated artifacts are excluded from GitHub through `.gitignore`, including:

- Raw or processed document collections where applicable
- BERT checkpoint weights
- Gemini embedding caches
- Chroma vector stores
- API caches
- Temporary experiment runs
- Other large, regenerable artifacts

The repository retains smaller research records where practical, including configurations, manifests, metrics, training histories, evaluation tables, and frozen predictions.

Consequently, a fresh clone can inspect the implementation, execute lightweight tests, and run analyses supported by the committed artifacts. Operations requiring excluded files need the complete local research environment or regenerated artifacts at the documented locations.

Exact hosted-model predictions may not be identical if Gemini evaluation is rerun later. The frozen predictions represent the submitted experimental record.

See [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md) for artifact descriptions, expected paths, provenance, loading procedures, and regeneration requirements.

## Testing

From `MVP/`, run:

```bash
pytest tests/
```

At the time of final verification:

```text
85 tests passed
```

The test suite verifies that:

- Default CLI modes are read-only and do not access the network
- Recomputed metrics match the submitted results
- Future runs cannot overwrite frozen artifacts
- Artifact schemas remain valid
- Every pipeline stage appears in CLI help
- Sibling-directory path resolution and `NEWSTART_BENCHMARK_ROOT` overrides work
- The MVP does not import the original `newstart_ai` package
- Python files and notebook code cells contain no prohibited path injection

All seven primary research notebooks were also executed successfully from the relocated root-level MVP.

## Output Isolation

Submitted experimental results are treated as frozen artifacts. Future reruns write to:

```text
MVP/runs/<run-id>/
```

This separation prevents future experiments from silently replacing or modifying the results reported in the capstone paper.

## Scope and Limitations

This MVP demonstrates document classification and agency routing. It does not establish that downstream agency-specific guidance is factually correct, safe, complete, or suitable as legal, financial, immigration, tax, or benefits advice.

Additional limitations include:

- The IRS class is substantially smaller than the other agency classes.
- Large artifacts are not all distributed through GitHub.
- Hosted Gemini outputs may change across future reruns.
- RAG effectiveness depends on the available training corpus and retrieval configuration.
- The study evaluates four agencies and should not automatically be generalized to other government domains.
- The test set contains 99 eligible, family-disjoint documents.
- The interactive application demonstrates feasibility; its guidance quality was not formally evaluated.

## Authors and Contributions

NewStart AI was developed collaboratively by **Group 3**:

- [Ali Azizi](https://github.com/al1az1z1)
- [Cla-Petra Omaku](https://github.com/comaku-coder)
- [Jeffi Thomas](https://github.com/jeffiThomas)

The `MVP/` directory provides a cleaned, independently executable organization of the team’s final research workflow. It preserves the project’s shared methodology, implementation, experiments, and results and should not be interpreted as the work of a single contributor.

## Academic Information

**Institution:** University of San Diego  
**Program:** Master of Science in Applied Artificial Intelligence  
**Course:** AAI-590 — Capstone Project  
**Term:** Summer 2026  
**Professor:** [Anna Marbut](https://github.com/amarbut)

## AI-Assisted Development Disclosure

AI-assisted tools, including ChatGPT, Claude, and GitHub Copilot, were used to support selected project activities such as code organization, debugging, documentation refinement, language editing, and implementation review.

All AI-assisted output was reviewed, tested, adapted, and integrated by the project team. The team remained responsible for the research design, methodological decisions, data preparation, experiments, interpretation of results, verification, and final submitted work.

AI tools were used to enhance the development and learning process rather than replace the team’s understanding or academic responsibility.

## Documentation

- [Artifact documentation](docs/ARTIFACTS.md)
- [Pipeline stages](docs/STAGES.md)
- [Safe rerun guide](docs/RERUN_GUIDE.md)
- [Final model comparison](notebooks/07_final_comparison.ipynb)
- [BERT error attribution](notebooks/06_bert_error_attribution.ipynb)

## Repository

The complete project repository is available at:

[github.com/al1az1z1/newstart-ai](https://github.com/al1az1z1/newstart-ai)

For the capstone research implementation, begin with this `MVP/` directory. For the broader application, development history, data-acquisition work, and experimental workspace, see [`newstart_ai_benchmark/`](../newstart_ai_benchmark/).
