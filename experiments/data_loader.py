from datasets import load_dataset

def load_xsum_dataset(sample_size=100):
    print("Loading XSUM dataset...")
    
    dataset = load_dataset("xsum")
    
    # Use small subset (important for speed)
    data = dataset["train"].select(range(sample_size))
    
    print(f"Loaded {len(data)} samples")
    
    return data


def format_input(example):
    return f"Summarize the following document:\n{example['document']}"


if __name__ == "__main__":
    data = load_xsum_dataset()
    
    print("\nSample Raw Data:")
    print(data[0])
    
    print("\nFormatted Input:")
    formatted = format_input(data[0])
    print(formatted[:500])