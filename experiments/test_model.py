from experiments.data_loader import load_xsum_dataset, format_input
from models.flan_t5 import FlanT5Model

if __name__ == "__main__":
    # Load data
    data = load_xsum_dataset(sample_size=3)
    
    # Load model
    model = FlanT5Model()
    
    # Test on one sample
    sample = data[0]
    input_text = format_input(sample)
    
    print("\nGenerating output...\n")
    
    output = model.generate(input_text)
    
    print("Model Output:")
    print(output)
    
    print("\nActual Summary:")
    print(sample["summary"])