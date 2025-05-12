import argparse
from models.t5_model import T5Summarizer
from models.bart_model import BARTSummarizer
from models.t5_long_model import LongT5Summarizer

def read_sample(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def save_summary(text, model_name):
    output_file = f"results/samenvatting_{model_name}_3.txt"
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(text)
    print(f"Samenvatting opgeslagen in {output_file}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Samenvatten van tekst met verschillende modellen.")
    parser.add_argument("--model", type=str, choices=["t5", "bart", "t5long"], required=True, help="Kies het model: t5, t5long of bart")
    args = parser.parse_args()

    # Inladen Tekst
    tekst = read_sample("text_samples/voorbeeld1.txt")

    # Model kiezen
    if args.model == "t5":
        model = T5Summarizer()
    elif args.model == "bart":
        model = BARTSummarizer()
    elif args.model == "t5long":
        model = LongT5Summarizer()
    else:
        raise ValueError("Ongeldig model gekozen. Kies 't5' of 'bart'.")
    
    # Samenvatten
    samenvatting = model.summarize(tekst)

    # Output opslaan
    save_summary(samenvatting, args.model)