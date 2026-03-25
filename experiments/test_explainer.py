from experiments.data_loader import load_xsum_dataset, format_input
from mexgen.explainer import MexGenExplainer
from experiments.visualize import plot_importance
from experiments.evaluation import compute_faithfulness, compute_cost
from mexgen.perturbation import remove_sentence
from experiments.visualize import plot_faithfulness_vs_cost

if __name__ == "__main__":
    # Load dataset (multiple samples)
    data = load_xsum_dataset(sample_size=3)

    # Initialize explainer once
    explainer = MexGenExplainer()

    # Loop through all samples
    for i, sample in enumerate(data):
        print(f"\n\n================ SAMPLE {i} ================\n")

        input_text = format_input(sample)

        # Run explanation
        original_output, scores = explainer.explain(input_text)

        TOP_K = 5

        print("\n=== TOP IMPORTANT SENTENCES ===\n")

        for idx, sentence, raw, norm in scores[:TOP_K]:
            print(f"Sentence {idx}")
            print(f"Raw Importance: {raw:.4f}")
            print(f"Normalized Importance: {norm:.4f}")
            print(sentence)
            print("-" * 50)

        # Plot only for first sample (to avoid multiple popups)
        if i == 0:
            plot_importance(scores)

        # ===============================
        # EVALUATION (FAITHFULNESS)
        # ===============================

        top_idx, _, raw, norm = scores[0]

        print("\n=== FAITHFULNESS TEST ===\n")

        faithfulness, new_output = compute_faithfulness(
            explainer.model,
            input_text,
            original_output,
            top_idx,
            remove_sentence
        )

        print(f"Most Important Sentence Index: {top_idx}")
        print(f"Faithfulness Score: {faithfulness:.4f}")

        print("\nNew Output after removing important sentence:")
        print(new_output)

        # ===============================
        # COST
        # ===============================

        num_sentences = len(scores)
        cost = compute_cost(num_sentences)

        print("\n=== COST ===")
        print(f"Total Model Calls: {cost}")
        print("\n=== RESULT SUMMARY ===")
        print(f"Top Importance: {raw:.4f}")
        print(f"Faithfulness: {faithfulness:.4f}")
        print(f"Cost: {cost}")
        plot_faithfulness_vs_cost(cost, faithfulness)