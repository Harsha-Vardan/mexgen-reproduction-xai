# MExGen Reproduction for Explainable AI (XAI)

## 📌 Project Overview

This project reproduces and implements the core ideas of the **MExGen (Multi-level Explanations for Generative Models)** framework.

The goal is to explain **why a Large Language Model (LLM)** generates a particular output by identifying **which parts of the input context influence the response the most**.

---

## ❗ Problem Statement

Traditional Explainable AI methods like **LIME** and **SHAP** work well for models with **scalar outputs**, but fail for **generative models** because:

- LLMs produce **long textual outputs**
- There is **no direct scalar output**
- Input context can be **very large**

---

## 💡 Our Approach (MExGen)

We solve this using the MExGen framework:

### 🔹 1. Scalarization using Similarity
We convert text outputs into scalar values using **BERTScore**:
- Compare original output with perturbed outputs
- Generate a similarity score

### 🔹 2. Perturbation-based Explanation
- Remove one sentence at a time from input
- Observe how output changes
- Measure importance using:

`Importance = 1 - similarity`

### 🔹 3. Multi-Level Explanation (Concept)
- Coarse level → paragraph
- Fine level → sentence (implemented)
- Future extension → word level

---

## 🔄 Workflow

1. Input document + query  
2. Generate output using LLM (FLAN-T5)  
3. Create perturbed inputs (remove sentences)  
4. Generate new outputs  
5. Compute similarity scores (BERTScore)  
6. Calculate importance values  
7. Rank sentences based on importance  
8. Evaluate using faithfulness and cost  
9. Visualize importance  

---

## 🧠 Key Idea

> If removing a part of the input significantly changes the output, that part is important.

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- FLAN-T5 (LLM)
- BERTScore
- Matplotlib
- NLTK

---

## 📂 Project Structure
mexgen-reproduction-xai/
│
├── experiments/
│ ├── data_loader.py
│ ├── test_explainer.py
│ ├── evaluation.py
│ ├── visualize.py
│
├── mexgen/
│ ├── explainer.py
│ ├── perturbation.py
│ ├── scalarizer.py
│
├── models/
│ ├── flan_t5.py
│
├── README.md
└── requirements.txt


---

## ⚙️ Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/Harsha-Vardan/mexgen-reproduction-xai.git
cd mexgen-reproduction-xai
2. Install Dependencies
pip install -r requirements.txt
3. Run the Project
python -m experiments.test_explainer
🔍 What the Code Does

For each input sample:

Generate original output using FLAN-T5
Remove each sentence one by one
Generate new outputs
Compute similarity using BERTScore
Calculate importance scores
Rank sentences
Evaluate explanation:
Faithfulness
Cost
Visualize importance
📊 Evaluation Metrics
🔹 Faithfulness

Measures how much output changes when important input is removed:

Faithfulness = 1 - similarity

Higher value → better explanation

🔹 Cost

Measures computational expense:

Cost = 1 + number of sentences

📊 Results Summary
Metric	Value (Typical)
Top Importance	~0.10
Faithfulness	~0.10
Cost	~20–25 model calls

👉 These results indicate that the model relies on distributed contextual information, and removing individual sentences leads to moderate changes in output.

🧪 Example Explanation Output
Sentence 4
Raw Importance: 0.1041
Normalized Importance: 1.0000
"First Minister Nicola Sturgeon visited the area..."

Sentence 5
Raw Importance: 0.1022
Normalized Importance: 0.9816
"The waters breached a retaining wall..."
📈 Visualization
🔹 Sentence Importance Graph

🔹 Faithfulness vs Cost

(Add screenshots in the assets/ folder)

🧠 Key Observations
The model relies on distributed context
No single sentence dominates output
Some irrelevant sentences receive importance due to semantic overlap
Faithfulness scores indicate robust summarization behavior
🔬 Extension (Our Contribution)

We extend the MExGen framework to explore its applicability in:

Offline RAG (Retrieval-Augmented Generation) systems
Language-based generative tasks
Evaluating whether explanation fidelity holds in long-context settings

🚀 Contributions
Implemented full MExGen pipeline
Adapted LIME/SHAP ideas to generative models
Built sentence-level explanation system
Added normalization and visualization
Evaluated using faithfulness and cost
Extended to multi-sample evaluation

🔮 Future Work
Paragraph-level hierarchical refinement
Word-level explanations
Real-time explanation dashboards
Integration with production LLM systems

🧾 Reproducibility
Random Seed: 42
Dataset: XSUM (Hugging Face)

👨‍💻 Author
Harsha Vardan

📌 Conclusion

This project demonstrates how explainability can be extended to generative language models by transforming textual outputs into scalar similarity scores and analyzing the influence of input components using perturbation-based techniques.


---

