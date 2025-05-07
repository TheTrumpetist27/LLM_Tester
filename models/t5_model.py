from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class T5Summarizer:
    def __init__(self, model_name="t5-large"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def summarize(self, text, max_length=150, min_length=40):
        input_text = "Please write a summary of the following article text: " + text.strip().replace("\n", " ")
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        summary_ids = self.model.generate(inputs["input_ids"], 
                                          max_length=max_length, 
                                          min_length=min_length, 
                                          length_penalty=2.0, 
                                          num_beams=4, 
                                          early_stopping=True)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)