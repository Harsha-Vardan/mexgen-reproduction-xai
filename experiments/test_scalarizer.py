from mexgen.scalarizer import compute_similarity

if __name__ == "__main__":
    original = "Flooding caused major damage in Scotland."
    perturbed = "Heavy rain caused flooding and damage."
    
    sim = compute_similarity(original, perturbed)
    
    print("Similarity Score:", sim)