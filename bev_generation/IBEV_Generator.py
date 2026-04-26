from abc import ABC, abstractmethod
import torch


class IBEVGenerator(ABC):
    def __init__(self, model_path: str = None, device='cuda', use_eval=True):
        super().__init__()
        self.generator = None

    @abstractmethod
    def infer(self, obs_dict):
        """
        No-gradient inference. Returns binarized BEV tensor (B, C, H, W) with values in {0, 255}.
        """
        pass

    @abstractmethod
    def forward_train(self, obs_dict) -> torch.Tensor:
        """
        Forward pass with gradients. Returns raw (non-binarized) model output.
        Use for training; do not wrap in torch.no_grad().
        """
        pass

    @abstractmethod
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Architecture-specific loss between raw prediction and normalized target in [0, 1].
        - UNet:    BCEWithLogitsLoss (pred = raw logits)
        - CVT 3ch: L1Loss (pred = sigmoid output)
        - CVT 6ch: BCELoss on first 3 channels (pred = sigmoid output)
        """
        pass

    def set_train(self):
        """Switch internal generator to training mode."""
        self.generator.train()

    def set_eval(self):
        """Switch internal generator to eval mode."""
        self.generator.eval()

    def parameters(self):
        """Expose generator parameters for an external optimizer."""
        return self.generator.parameters()

