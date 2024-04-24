import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from tqdm import tqdm

from autoencoder_multilayer import AutoEncoderMultiLayer
from transition_model import FeatureTransitionModel

dataset_name = "wikitext"
dataset_config = "wikitext-103-v1"
dataset_split = "train"
model_name = "gpt2"
num_layers = 12
batch_size = 16
total_samples = 50000
steps_per_report = 100
dtype = torch.bfloat16

dataset = load_dataset(dataset_name, dataset_config, split=dataset_split).shuffle()
model = HookedTransformer.from_pretrained(model_name, device="cuda", dtype=dtype)
autoencoder = AutoEncoderMultiLayer.load("autoencoder", "gpt2_wikitext_16x_v2")
transition_model = FeatureTransitionModel(autoencoder.cfg.m_dim, num_layers).to(dtype=dtype, device="cuda")

try:
    loss_acc = 0

    for i in tqdm(range(total_samples // batch_size)):
        act_names = [f"blocks.{l}.hook_mlp_out" for l in range(num_layers)]

        seqs = dataset[batch_size * i:(batch_size * i) + batch_size]["text"]
        seqs = [s for s in seqs if len(s) > 0]
        if len(seqs) == 0:
            continue

        torch.cuda.empty_cache()

        tokens = model.tokenizer(seqs, return_tensors="pt", truncation=True, padding=True, max_length=192)["input_ids"]

        output, cache = model.run_with_cache(tokens, names_filter=act_names)

        # [num_layers, batch_dim, seq_length, n_dim]
        mlp_outs = torch.stack([cache[act_name] for act_name in act_names])

        # [seq_length*batch_dim, num_layers, n_dim]
        mlp_outs = mlp_outs.permute(2, 1, 0, 3).reshape(-1, num_layers, mlp_outs.shape[-1])

        features = autoencoder.encode(mlp_outs).swapaxes(0, 1).detach()

        del mlp_outs, output, cache
        torch.cuda.empty_cache()

        loss = transition_model.train_on(features)

        loss_acc += loss

        if i % steps_per_report == 0 and i > 0:
            tqdm.write(f"Loss: {loss_acc / steps_per_report}")
            loss_acc = 0
finally:
    state = transition_model.state_dict()
    torch.save(state, "transition_model.pth")