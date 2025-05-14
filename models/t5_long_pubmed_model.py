from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class LongT5PubmedSummarizer:
    def __init__(self, model_name="Stancld/longt5-tglobal-large-16384-pubmed-3k_steps"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def summarize(self, text, max_length=512):
        input_text = "summarize: " + text.strip().replace("\n", " ")
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=16384).to(self.device)

        print(f"Aantal tokens: {inputs['input_ids'].shape[1]}")

        summary_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            length_penalty=1.0,
            num_beams=4,
            early_stopping=False,
            no_repeat_ngram_size=5
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)