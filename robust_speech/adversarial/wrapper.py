from typing import Any, TypeVar
import foolbox as fb 
import speechbrain as sb
import eagerpy as ep
T = TypeVar("T")

class BrainModel(fb.PyTorchModel):
    def __init__(
        self,
        model: sb.Brain,
        bounds: fb.types.BoundsInput,
        device: Any = None,
        preprocessing: fb.types.Preprocessing = None,
    ):
        import torch

        if not isinstance(model, sb.Brain):
            raise ValueError("expected model to be a sb.Brain instance")
        device = model.device
        dummy = torch.zeros(0, device=device)

        

        super().__init__(
            model, bounds=bounds, dummy=dummy, preprocessing=preprocessing
        )

        self.data_format = "batch_first"
        self.device = device

class BrainMisclassification(fb.criteria.Criterion):
    """Considers those perturbed inputs adversarial whose predicted class
    differs from the label.
    Args:
        labels: Tensor with labels of the unperturbed inputs ``(batch,)``.
    """

    def __init__(self, labels: Any):
        super().__init__()
        self.labels = labels

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.labels!r})"

    def __call__(self, perturbed: T, outputs: T) -> T:
        outputs_, restore_type = ep.astensor_(outputs)
        del perturbed, outputs

        classes = outputs_.argmax(axis=-1)
        is_adv = classes != self.labels
        return restore_type(is_adv)
