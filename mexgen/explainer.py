from mexgen.perturbation import generate_perturbations, get_sentences
from mexgen.scalarizer import compute_similarity
from models.flan_t5 import FlanT5Model

def normalize_scores(scores):
    """
    Normalize importance scores to [0,1]
    Keeps relative differences for visualization
    """
    values = [s[2] for s in scores]

    min_val = min(values)
    max_val = max(values)

    normalized = []

    for idx, sentence, score in scores:
        if max_val - min_val == 0:
            norm = 0
        else:
            norm = (score - min_val) / (max_val - min_val)

        normalized.append((idx, sentence, score, norm))

    return normalized


class MexGenExplainer:
    def __init__(self):
        self.model = FlanT5Model()

    def explain(self, input_text):
        print("Generating original output...")
        original_output = self.model.generate(input_text)

        print("\nOriginal Output:")
        print(original_output)

        # Get sentences
        sentences = get_sentences(input_text)

        # Generate perturbed inputs
        perturbations = generate_perturbations(input_text)

        importance_scores = []

        print("\nComputing importance for each sentence...\n")

        for idx, perturbed_text in perturbations:
            print(f"Removing sentence {idx}...")

            perturbed_output = self.model.generate(perturbed_text)

            similarity = compute_similarity(original_output, perturbed_output)

            importance = 1 - similarity

            importance_scores.append((idx, sentences[idx], importance))

        # Sort by importance
        importance_scores.sort(key=lambda x: x[2], reverse=True)

        # Normalize
        normalized_scores = normalize_scores(importance_scores)

        return original_output, normalized_scores  