import numpy as np
import torch
import clip
import torchvision
from utils.Roberta import RoBERTaClassifier
from utils.ImageOnly import Decoder4
from utils.CustomDataset import CustomDataset
from utils.test import test
from transformers import RobertaTokenizer


def get_emotions(image, text):
    tags = ['Excitement', 'Sadness', 'Amusement', 'Disgust', 'Awe', 'Contentment', 'Fear', 'Anger']
    max_len = 128
    input_dim = 768
    output_dim = 8
    print(image)

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    test_emo = np.zeros((1, 8))
    text = [text]

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    test_dataset = CustomDataset(image, text, test_emo, tokenizer, max_len, test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False, num_workers=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)

    model2 = RoBERTaClassifier(num_labels=output_dim)
    decoder = Decoder4(input_dim, output_dim).to(device)
    model2.load_state_dict(torch.load('models/Roberta.pth', map_location=device))
    decoder.load_state_dict(torch.load('models/Custom.pth', map_location=device))
    decoder = decoder.to(device)

    y_pred = test(model, model2, decoder, device, test_loader)
    del model, model2, decoder, test_loader
    torch.cuda.empty_cache()
    pred = y_pred.flatten()

    return pred
