from mexgen.perturbation import generate_perturbations

if __name__ == "__main__":
    text = "This is sentence one. This is sentence two. This is sentence three."
    
    perturbed = generate_perturbations(text)
    
    for idx, p in perturbed:
        print(f"\nRemoved sentence {idx}:")
        print(p)