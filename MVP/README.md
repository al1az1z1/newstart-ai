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

This `MVP/` directory is the recommended starting point for professors, reviewers, researchers, and users interested in the completed capstone study. It corresponds to the submitted research report, presentation, methodology, and frozen experimental results.

The sibling [`newstart_ai_benchmark/`](../newstart_ai_benchmark/) directory contains the broader application and development workspace, including the original experiments, data-acquisition work, application components, datasets, and research artifacts.

---

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

Under the primary `complete_unmasked` condition, all three methods achieved:

| Method | Accuracy | Macro F1 |
|---|---:|---:|
| Fine-tuned BERT | 1.000 | 1.000 |
| Gemini | 1.000 | 1.000 |
| Gemini + RAG | 1.000 | 1.000 |

The remaining nine conditions evaluate robustness when only certain document regions are available or when explicit agency identifiers, form numbers, OMB numbers, and government URLs are masked.

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
| Learning rate | \(2 \times 10^{-5}\) |
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
- Top-\(k = 3\) retrieval
- Approximately 8,600 indexed training chunks
- No test documents in the retrieval indexes

## MVP Architecture

`MVP/` is a self-contained Python package. The implementation under `newstart_ai_mvp/` contains its own simplified and reorganized copy of the required research pipeline logic.

It does not import executable Python code from:

```text
newstart_ai_benchmark/src/newstart_ai/