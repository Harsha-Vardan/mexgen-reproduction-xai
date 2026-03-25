# MExGen Reproduction for Explainable AI (XAI)

## Project Overview

This repository reproduces and implements the core ideas of the MExGen
(Multi-level Explanations for Generative Models) framework.

The goal is to explain why a Large Language Model (LLM) generates a
particular output by identifying which parts of the input context influence
the response most.

## Problem Statement

Traditional Explainable AI methods such as LIME and SHAP are designed for
models with scalar outputs. They are not directly suitable for generative
models because:

- LLMs produce long textual outputs.
- There is no direct scalar output to attribute.
- Input context can be very large.

## Approach (MExGen)

### 1. Scalarization Using Similarity

Text outputs are converted into scalar values using BERTScore:

- Compare original output with perturbed outputs.
- Generate a similarity score.

### 2. Perturbation-Based Explanation

- Remove one sentence at a time from the input.
- Observe how the output changes.
- Measure sentence importance using:

`importance = 1 - similarity`

### 3. Multi-Level Explanation (Concept)

- Coarse level: paragraph
- Fine level: sentence (implemented)
- Future extension: word level

## Workflow

1. Input document and query
2. Generate output using FLAN-T5
3. Create perturbed inputs (remove one sentence at a time)
4. Generate new outputs for each perturbation
5. Compute similarity scores (BERTScore)
6. Calculate importance values
7. Rank sentences by importance
8. Evaluate explanation quality using faithfulness and cost
9. Visualize importance

## Key Idea

If removing a part of the input significantly changes the output, that part is
important.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- FLAN-T5
- BERTScore
- Matplotlib
- NLTK

## Project Structure

```text
mexgen-reproduction-xai/
|-- experiments/
|   |-- data_loader.py
|   |-- test_explainer.py
|   |-- evaluation.py
|   `-- visualize.py
|-- mexgen/
|   |-- explainer.py
|   |-- perturbation.py
|   `-- scalarizer.py
|-- models/
|   `-- flan_t5.py
|-- README.md
`-- requirements.txt
```

## Setup Instructions

1. Clone the repository.

```bash
git clone https://github.com/Harsha-Vardan/mexgen-reproduction-xai.git
cd mexgen-reproduction-xai
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Run the pipeline.

```bash
python -m experiments.test_explainer
```

## What the Code Does

For each input sample, the pipeline:

- Generates an original output using FLAN-T5.
- Removes each sentence one by one.
- Regenerates outputs for perturbed inputs.
- Computes similarity with BERTScore.
- Calculates and ranks importance scores.
- Evaluates explanations with faithfulness and cost.
- Produces visualizations.

## Evaluation Metrics

### Faithfulness

Measures how much the output changes when important input is removed:

`faithfulness = 1 - similarity`

Higher values indicate stronger attribution effects.

### Cost

Measures computational effort:

`cost = 1 + number_of_sentences`

## Typical Results

| Metric         | Typical Value      |
| -------------- | ------------------ |
| Top importance | ~0.10              |
| Faithfulness   | ~0.10              |
| Cost           | ~20-25 model calls |

These results suggest the model relies on distributed contextual information,
so removing single sentences causes moderate changes rather than drastic shifts.

## Example Explanation Output

```text
Sentence 4
Raw Importance: 0.1041
Normalized Importance: 1.0000
"First Minister Nicola Sturgeon visited the area..."

Sentence 5
Raw Importance: 0.1022
Normalized Importance: 0.9816
"The waters breached a retaining wall..."
```

## Visualizations

- Sentence importance graph
- Faithfulness vs. cost graph

You can add generated screenshots to an assets folder and reference them here.

## Key Observations

- The model relies on distributed context.
- No single sentence dominates output generation.
- Some semantically overlapping sentences can receive non-trivial importance.
- Faithfulness values suggest robust summarization behavior.

## Extension (Contribution)

This reproduction extends MExGen exploration to:

- Offline RAG (Retrieval-Augmented Generation) settings
- Language-focused generative tasks
- Long-context scenarios where explanation fidelity may degrade

## Contributions

- Implemented the full MExGen pipeline
- Adapted LIME/SHAP-style ideas to generative models
- Built sentence-level explanation workflow
- Added normalization and visualization
- Evaluated with faithfulness and cost
- Extended to multi-sample analysis

## Future Work

- Paragraph-level hierarchical refinement
- Word-level explanation granularity
- Real-time explanation dashboards
- Integration with production LLM systems

## Reproducibility

- Random seed: 42
- Dataset: XSUM (Hugging Face)

## Author

Harsha Vardan

## Conclusion

This project demonstrates how explainability can be extended to generative
language models by converting textual outputs into scalar similarity scores and
using perturbation-based analysis to estimate the influence of each input part.
