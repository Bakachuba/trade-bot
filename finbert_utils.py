# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# from typing import Tuple
#
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
#
# tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
# model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
# labels = ["positive", "negative", "neutral"]
#
#
# def estimate_sentiment(news):
#     if news:
#         tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)
#
#         result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])[
#             "logits"
#         ]
#         result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
#         probability = result[torch.argmax(result)]
#         sentiment = labels[torch.argmax(result)]
#         return probability, sentiment
#     else:
#         return 0, labels[-1]
#
#
# if __name__ == "__main__":
#     tensor, sentiment = estimate_sentiment(
#         ['markets responded positively to the news!', 'traders were pleasantly surprised!'])
#     """
#     (.venv) PS E:\Pycharm_projects\test_trade_bot> python .\finbert_utils.py
#     tensor(0.8979, grad_fn=<SelectBackward0>) positive
#     False
#     """
#
#     # tensor, sentiment = estimate_sentiment(['markets responded negatively to the news!','traders were displeased!'])
#     """
#     config.json: 100%|████████████████████████████████████████████████████████| 758/758 [00:00<00:00, 759kB/s]
#     vocab.txt: 100%|███████████████████████████████████████████████████████| 232k/232k [00:00<00:00, 1.25MB/s]
#     special_tokens_map.json: 100%|███████████████████████████████████████████████████| 112/112 [00:00<?, ?B/s]
#     pytorch_model.bin: 100%|███████████████████████████████████████████████| 438M/438M [01:19<00:00, 5.53MB/s]
#     tensor(0.9979, grad_fn=<SelectBackward0>) negative
#     False
#     model.safetensors: 100%|███████████████████████████████████████████████| 438M/438M [01:18<00:00, 5.59MB/s]
#     """
#
#     print(tensor, sentiment)
#     print(torch.cuda.is_available())

import torch
print(torch.cuda.is_available())        # Должно быть True
print(torch.cuda.get_device_name(0))    # GTX 1060
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
