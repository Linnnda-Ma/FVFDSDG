import torch
import torch.nn as nn
import torch.nn.functional as F


class VerticalTokenMixupLayer(nn.Module):
    """
    A simplified Vertical Token Mixup layer that does not use attention mechanisms.
    """

    def __init__(self, layer_index, apply_layers, kappa, vtm_attn_dim, vtm_stopgrad=False, has_cls=True, report_every=50, verbose=True):
        super(VerticalTokenMixupLayer, self).__init__()

        assert isinstance(apply_layers, list), "apply_layers should be a list of integers"
        assert 0 not in apply_layers, "VTM cannot be applied to the first layer (layer index 0)"
        
        # Configuration and initialization
        self.apply_layers = apply_layers
        self.layer_index = layer_index
        self.kappa = kappa
        self.vtm_stopgrad = vtm_stopgrad
        self.has_cls = has_cls
        self.report_every = report_every
        self.verbose = verbose

        self.step_count = 0
        self.warnings_count = 0

        # Initialize a linear layer as a replacement for attention
        self.linear = nn.Linear(vtm_attn_dim, vtm_attn_dim)

        self.reset()

    def reset(self):
        self.memory = {}

    def register_memory(self, src, saliency):
        saliency = saliency.data
        B, H, Qdim, Kdim = saliency.shape

        # Compute saliency and get top-k salient tokens
        saliency = saliency.mean(1)[:, 0, 1:] if self.has_cls else saliency.mean(dim=[1, 2])
        _, top_idx = torch.topk(saliency, k=self.kappa, dim=1, largest=True)
        salient_tokens = torch.gather(src, 1, top_idx.unsqueeze(-1).expand(-1, -1, src.shape[-1]))

        if str(self.layer_index) in self.memory and self.warnings_count < 8:
            print("WARNING: Overwriting previous layer vertical tokens.")
            self.warnings_count += 1

        self.memory[str(self.layer_index)] = salient_tokens

    def get_memory_tokens(self):
        return list(self.memory.values())

    def report_status(self, vtm_token_num, device):
        if self.verbose and self.step_count % self.report_every == 1:
            print(f"\n----------------------------------VTM Stats-----------------------------------")
            print(f"  Layer : {self.layer_index+1}\tDevice : {device}")
            print(f"     [kappa = {self.kappa}]")
            print(f"     {vtm_token_num} Multi-scale Tokens were Cross-Attended!")
            print(f"------------------------------------------------------------------------------")

    def forward(self, src, *args, **kwargs):
        if self.training:
            if self.layer_index in self.apply_layers:
                self.step_count += 1
                vtm_memories = self.get_memory_tokens()
                vtm_tokens = torch.stack(vtm_memories)
                L, B, T, D = vtm_tokens.shape
                vtm_tokens = vtm_tokens.transpose(0, 1).reshape(B, -1, D)
                vtm_tokens = vtm_tokens.detach() if self.vtm_stopgrad else vtm_tokens

                # Instead of attention, apply a simple linear transformation to src and VTM tokens
                combined_tokens = torch.cat([src, vtm_tokens], dim=1)  # Concatenate original and VTM tokens
                out = self.linear(combined_tokens)

                self.report_status(vtm_tokens.shape[1], out.device)

                if self.layer_index < max(self.apply_layers):
                    self.register_memory(src, out)

                return out
            else:
                out = self.linear(src)  # Apply linear transformation for non-applied layers
                if self.layer_index < max(self.apply_layers):
                    self.register_memory(src, out)
                return out
        else:
            if self.layer_index in self.apply_layers:
                return self.linear(src)  # Only linear transformation during evaluation
            else:
                return self.linear(src)
