import torch

def get_time_ids(batch):
    bsz = len(batch["caption"])
    

    h = batch["original_size"][0]
    w = batch["original_size"][1]
    cx = batch["crop_coords"][0]
    cy = batch["crop_coords"][1]
    th = batch["target_size"][0]
    tw = batch["target_size"][1]

    time_ids = []
    for i in range(bsz):
        time_ids.append([h[i].item(), w[i].item(), cx[i].item(), cy[i].item(), th[i].item(), tw[i].item()])

    return torch.tensor(time_ids)


def encode_prompt(batch, tokenizer1, tokenizer2, text_encoder1, text_encoder2, device):
    prompts = batch["caption"]

    tokens1 = tokenizer1(
        prompts,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    tokens2 = tokenizer2(
        prompts,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    emb1 = text_encoder1(**tokens1).last_hidden_state
    out2 = text_encoder2(**tokens2, output_hidden_states=True)
    emb2 = out2.hidden_states[-2]
    pooled_emb = out2.text_embeds

    prompt_embeds = torch.cat([emb1, emb2], dim=-1)
    return prompt_embeds.to(dtype=torch.bfloat16), pooled_emb.to(dtype=torch.bfloat16)