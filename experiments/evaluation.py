from mexgen.scalarizer import compute_similarity


def compute_faithfulness(model, input_text, original_output, top_sentence_idx, perturb_fn):
    """
    Remove most important sentence and check output change
    """
    perturbed_text = perturb_fn(input_text, top_sentence_idx)

    new_output = model.generate(perturbed_text)

    similarity = compute_similarity(original_output, new_output)

    faithfulness = 1 - similarity

    return faithfulness, new_output


def compute_cost(num_sentences):
    """
    Total model calls used
    """
    return 1 + num_sentences