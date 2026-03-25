from bert_score import score

def compute_similarity(original, perturbed):
    """
    Compute similarity between two outputs using BERTScore
    """
    P, R, F1 = score([perturbed], [original], lang="en", verbose=False)
    return F1.item()