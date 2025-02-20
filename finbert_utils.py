from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple

device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]


def estimate_sentiment(news):
    if news:
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)

        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])[
            "logits"
        ]
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
        probability = result[torch.argmax(result)]
        sentiment = labels[torch.argmax(result)]
        return probability, sentiment
    else:
        return 0, labels[-1]


if __name__ == "__main__":
    tensor, sentiment = estimate_sentiment(
        ['markets responded positively to the news!', 'traders were pleasantly surprised!'])

    """
    tensor(0.8979, device='cuda:0', grad_fn=<SelectBackward0>) positive
    True
    """

    # tensor, sentiment = estimate_sentiment(['markets responded negatively to the news!','traders were displeased!'])
    """
    tensor(0.9979, device='cuda:0', grad_fn=<SelectBackward0>) negative
    True
    """

    print(tensor, sentiment)
    print(torch.cuda.is_available())

# import torch
# print(torch.cuda.is_available())        # Должно быть True
# print(torch.cuda.get_device_name(0))    # GTX 1060
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
