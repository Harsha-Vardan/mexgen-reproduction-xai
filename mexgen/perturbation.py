import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def get_sentences(text):
    return sent_tokenize(text)


def remove_sentence(text, index):
    sentences = sent_tokenize(text)
    
    new_sentences = sentences[:index] + sentences[index+1:]
    
    return " ".join(new_sentences)


def generate_perturbations(text):
    sentences = sent_tokenize(text)
    
    perturbed_texts = []
    
    for i in range(len(sentences)):
        new_text = remove_sentence(text, i)
        perturbed_texts.append((i, new_text))
    
    return perturbed_texts