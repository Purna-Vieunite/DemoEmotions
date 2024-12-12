import torch
from transformers import AutoModel, AutoTokenizer


def get_caption(image):
    print(image)
    model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
    model = model.to(device='cuda')

    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
    model.eval()
    question = "Describe the image."
    msgs = [{'role': 'user', 'content': question}]

    res = model.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.7,
        stream=True
    )
    generated_text = ""
    for new_text in res:
        generated_text += new_text

    model.cpu()
    del model
    torch.cuda.empty_cache()
    return generated_text
