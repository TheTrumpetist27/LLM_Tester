from transformers import AutoTokenizer, LongT5ForConditionalGeneration
import torch

class LongT5BookSummarizer:
    def __init__(self, model_name="pszemraj/long-t5-tglobal-base-16384-book-summary"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = LongT5ForConditionalGeneration.from_pretrained(model_name)

    def summarize(self, text, max_length=512, min_length=8):
        text2 = (
            "In recent years, the field of artificial intelligence has made remarkable progress, "
            "especially in the area of natural language processing. Large language models such as GPT and T5 have revolutionized..."
        ) 
        input_text = text2.strip().replace("\n", " ")
        inputs = self.tokenizer(input_text, return_tensors="pt", padding="longest", truncation=False)
        print(f"Aantal tokens: {len(inputs['input_ids'][0])}")

        summary_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=0.3,
            #repetition_penalty=3.5,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            #encoder_no_repeat_ngram_size=3,
        )

        print(f"Generated IDs: {summary_ids}")
        print(f"Shape: {summary_ids.shape}")
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    #max_length=16384