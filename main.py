import argparse
from models.t5_model import T5Summarizer
from models.bart_model import BARTSummarizer
from models.t5_long_pubmed_model import LongT5PubmedSummarizer
from models.t5_long_book_model import LongT5BookSummarizer

def read_sample(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def save_summary(text, model_name):
    output_file = f"voorbeeldTweeResults/samenvatting_{model_name}_4.txt"
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(text)
    print(f"Samenvatting opgeslagen in {output_file}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Samenvatten van tekst met verschillende modellen.")
    parser.add_argument("--model", type=str, choices=["t5", "bart", "t5pubmed", "t5book"], required=True, help="Kies het model: t5, t5long of bart")
    args = parser.parse_args()

    # Inladen Tekst
    tekst = read_sample("text_samples/voorbeeld2.txt")

    # Model kiezen
    if args.model == "t5":
        model = T5Summarizer()
    elif args.model == "bart":
        model = BARTSummarizer()
    elif args.model == "t5pubmed":
        model = LongT5PubmedSummarizer()
    elif args.model == "t5book":
        model = LongT5BookSummarizer()
    else:
        raise ValueError("Ongeldig model gekozen.")
    
    # Samenvatten
    samenvatting = model.summarize(tekst)

    # Output opslaan
    save_summary(samenvatting, args.model)