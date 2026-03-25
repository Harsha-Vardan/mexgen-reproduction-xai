from experiments.data_loader import load_xsum_dataset, format_input
from mexgen.explainer import MexGenExplainer
from experiments.visualize import plot_importance


if __name__ == "__main__":
    data = load_xsum_dataset(sample_size=1)

    sample = data[0]
    input_text = format_input(sample)

    explainer = MexGenExplainer()

    original_output, scores = explainer.explain(input_text)

    TOP_K = 5

    print("\n\n=== TOP IMPORTANT SENTENCES ===\n")

    for idx, sentence, raw, norm in scores[:TOP_K]:
        print(f"Sentence {idx}")
        print(f"Raw Importance: {raw:.4f}")
        print(f"Normalized Importance: {norm:.4f}")
        print(sentence)
        print("-" * 50)

    plot_importance(scores)