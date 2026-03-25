from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class FlanT5Model:
    def __init__(self, model_name="google/flan-t5-base"):
        print("Loading FLAN-T5 model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        print("Model loaded successfully!")

    def generate(self, input_text, max_length=128):
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True)
        
        outputs = self.model.generate(**inputs, max_length=max_length)
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)