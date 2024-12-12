import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, image, texts, labels, tokenizer, max_len, transforms=None):
        self.image = image
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transforms = transforms

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        image = self.image
        text = str(self.texts[idx])
        label = self.labels[idx]
        if self.transforms:
            image = self.transforms(image)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.float),
            'images': image
        }
