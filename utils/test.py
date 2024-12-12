import numpy as np
import torch


def test(model, model2, decoder, device, test_loader):
    model = model.to(device)
    decoder = decoder.to(device)
    decoder.eval()
    model2 = model2.to(device)
    model2.eval()

    y_pred_val = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            images = batch['images'].to(device)
            outputs1 = model2(input_ids, attention_mask)
            with torch.no_grad():
                image_features = model.encode_image(images)
            image_features = image_features.to(torch.float32)
            outputs2 = decoder(image_features)

            outputs = (3 * outputs1 + 1 * outputs2) / 4

            preds = outputs
            y_pred_val.extend(preds.cpu().numpy())

    y_pred = np.array(y_pred_val)
    y_pred = np.reshape(y_pred, (-1, 8))

    model.cpu()
    model2.cpu()
    decoder.cpu()
    return y_pred
