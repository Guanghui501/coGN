"""
Training wrapper with contrastive learning support.

This module provides a wrapper around the standard training function
to support contrastive learning loss.
"""

import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    """Combined loss for task prediction and contrastive learning.

    Combines MSE loss for the main task with contrastive loss for
    representation learning.
    """

    def __init__(self, task_criterion, use_contrastive=False, contrastive_weight=0.1):
        """Initialize combined loss.

        Args:
            task_criterion: Loss function for main task (e.g., nn.MSELoss())
            use_contrastive: Whether to use contrastive loss
            contrastive_weight: Weight for contrastive loss term
        """
        super().__init__()
        self.task_criterion = task_criterion
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight

    def forward(self, output, target):
        """Compute combined loss.

        Args:
            output: Model output (dict if contrastive enabled, else tensor)
            target: Ground truth labels

        Returns:
            loss: Combined loss value
        """
        # Handle dict output (with contrastive loss)
        if isinstance(output, dict):
            predictions = output['predictions']
            task_loss = self.task_criterion(predictions, target)

            # Add contrastive loss if available
            if self.use_contrastive and 'contrastive_loss' in output:
                contrastive_loss = output['contrastive_loss']
                total_loss = task_loss + self.contrastive_weight * contrastive_loss
                return total_loss
            else:
                return task_loss
        else:
            # Standard case (no contrastive loss)
            return self.task_criterion(output, target)


def create_model_wrapper(model, use_contrastive=False):
    """Create a model wrapper that handles contrastive learning output.

    Args:
        model: The ALIGNN model
        use_contrastive: Whether contrastive learning is enabled

    Returns:
        Wrapped model function
    """
    def model_fn(batch):
        """Model forward function that handles dict/tensor output."""
        if use_contrastive:
            # Model will return dict with predictions and features
            output = model(batch)
            return output
        else:
            # Standard mode
            return model(batch)

    return model_fn
