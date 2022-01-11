import foolbox as fb 
import speechbrain as sb
from fb.models.base import Model
import eagerpy as ep

class L2PGD(fb.attacks.projected_gradient_descent.L2ProjectedGradientDescentAttack):
    def get_loss_fn(
        self, model: Model, labels: ep.Tensor
    ) -> Callable[[ep.Tensor], ep.Tensor]:
        # can be overridden by users
        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            predictions = model._model.compute_forward(inputs, sb.Stage.TRAIN)
            loss = model._model.compute_objectives(predictions, batch, sb.Stage.TRAIN)
            return loss

        return loss_fn