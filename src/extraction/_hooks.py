import torch

Module = torch.nn.Module
Tensor = torch.Tensor


class HookModel(Module):
    def __init__(
        self,
        model: Module,
        freeze: bool = False,
        feature_transform: callable = lambda x: x,
    ):
        """Transform is what we apply to the activations before storing them."""
        super().__init__()
        self.model = model
        self.activations = {}
        self.freeze = freeze
        self.feature_transform = feature_transform

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def register_feature_transform(self, transform: callable) -> None:
        self.feature_transform = transform

    def register_hook(self, module_name: str, requires_grad: bool = False) -> None:
        def named_hook(name):
            def hook(module, input, output):
                transformed = self.feature_transform(output)
                self.activations[name] = transformed

            return hook

        layers = dict([*self.model.named_modules()])
        if module_name not in layers:
            raise ValueError(f"Layer '{module_name}' is not found in the model.")

        module = layers[module_name]
        module.register_forward_hook(named_hook(module_name))

        if requires_grad:
            for param in module.parameters():
                param.requires_grad = True

    def forward(self, x: Tensor) -> Tensor:
        self.activations = {}
        logits = self.model(x)

        # this is for the case where the model returns multiple outputs (eg the case for clip)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        if not self.activations:
            raise RuntimeError("No activations were recorded during the forward pass.")

        activations = (
            next(iter(self.activations.values()))
            if len(self.activations) == 1
            else self.activations
        )

        return logits, activations
