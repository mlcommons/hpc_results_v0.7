"""SGD with Weight Decay"""

import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import training_ops

from typeguard import typechecked
from typing import Optional


@tf.keras.utils.register_keras_serializable(package="Addons")
class MomentumWD(tf.keras.optimizers.Optimizer):
    """SGD with Weight Decay
    """

    @typechecked
    def __init__(
        self,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        name: str = "MomentumWD",
        use_nesterov: bool = False,
        **kwargs
    ):
        """Construct a new LARS Optimizer.

      Args:
          learning_rate: A `Tensor` or floating point value. The base learning rate.
          momentum: A floating point value. Momentum hyperparameter.
          weight_decay: A floating point value. Weight decay hyperparameter. Adding an
             L2 regularizer to a Keras variable is not equivalent to how the LARS paper
             handles weight decay. In the LARS paper, when computing the "trust"
             coefficient, the magnitude of the gradient and the magnitude weights
             are added together. But if an L2 regularizer is added to a Keras variable,
             the gradient and weights are first added together, and then the magnitude is taken
          name: Optional name prefix for variables and ops created by LARS Optimizer.
          use_nesterov: when set to True, nesterov momentum will be enabled

      Raises:
          ValueError: If a hyperparameter is set to a non-sensical value.
      """
        if momentum < 0.0:
            raise ValueError("momentum should be positive: %s" % momentum)
        if weight_decay < 0.0:
            raise ValueError("weight_decay should be positive: %s" % weight_decay)
        super(MomentumWD, self).__init__(name=name, **kwargs)

        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("momentum", momentum)
        self._set_hyper("weight_decay", weight_decay)
        self.use_nesterov = use_nesterov

    def get_config(self):
        config = {
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "momentum": self._serialize_hyperparameter("momentum"),
            "weight_decay": self._serialize_hyperparameter("weight_decay"),
            "use_nesterov": self.use_nesterov,
        }
        base_config = super(MomentumWD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "momentum")

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype

        lr = self._get_hyper("learning_rate", var_dtype)
        weight_decay = self._get_hyper("weight_decay", var_dtype)
        mom = self.get_slot(var, "momentum")
        momentum = self._get_hyper("momentum", var_dtype)
        use_nesterov = self.use_nesterov

        # Add the weight regularization gradient
        grad = grad + weight_decay * var

        return training_ops.resource_apply_momentum(
            var.handle,
            mom.handle,
            math_ops.cast(1.0, var_dtype),
            grad * lr,
            momentum,
            use_locking=False,
            use_nesterov=use_nesterov,
        )

    def _resource_apply_sparse(self, grad, var, indices):
        mom = self.get_slot(var, "momentum")
        use_nesterov = self.use_nesterov
        return training_ops.resource_sparse_apply_momentum(
            var.handle,
            mom.handle,
            math_ops.cast(self._learning_rate_tensor, grad.dtype),
            grad,
            indices,
            math_ops.cast(self._momentum_tensor, grad.dtype),
            use_locking=False,
            use_nesterov=use_nesterov,
        )
