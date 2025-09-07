import torch
import torch.nn as nn
import torch.nn.functional as F

class PhoneticContextAwareLoss(nn.Module):
    """
    Implements the Phonetic Context-Aware Loss proposed in the paper:
    "Learning Phonetic Context-Dependent Viseme for Enhancing Speech-Driven 3D Facial Animation".

    This loss function calculates weights for each frame based on the magnitude of movement
    in the ground truth motion. It then applies these weights to the MSE loss, enabling
    the model to focus more on frames with significant motion.
    """
    def __init__(self, window_size: int = 5, reduction: str = 'mean'):
        """
        Args:
            window_size (int): The size of the window to consider for neighboring frames
                               when calculating weights. Must be an odd number.
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'.
        """
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError(f"window_size must be an odd number. Got: {window_size}")
        self.window_size = window_size
        self.reduction = reduction

    def forward(self, pred_motion: torch.Tensor, gt_motion: torch.Tensor) -> torch.Tensor:
        """
        Calculates the weighted Mean Squared Error loss.

        Args:
            pred_motion (torch.Tensor): The predicted motion tensor of shape (B, T, V*3).
            gt_motion (torch.Tensor): The ground truth motion tensor of shape (B, T, V*3).

        Returns:
            torch.Tensor: The calculated final loss value.
        """
        B, T, _ = gt_motion.shape

        # 1. Calculate the magnitude of movement between consecutive frames in the ground truth motion.
        # diff shape: (B, T-1, feat_dim)
        diff = gt_motion[:, 1:] - gt_motion[:, :-1]
        # movement shape: (B, T-1)
        movement = torch.norm(diff, p=2, dim=2)
        
        # Prepend a zero to represent no movement before the first frame.
        # This changes movement shape from (B, T-1) to (B, T).
        movement = F.pad(movement, (1, 0), 'constant', 0)

        # 2. Compute the average movement for each frame using a sliding window.
        # Add padding to handle boundaries during the windowing operation.
        padding = (self.window_size - 1) // 2
        # F.pad applies padding from the last dimension onwards. For a (B, T-1) tensor,
        # the padding is (left, right).
        movement_padded = F.pad(movement, (padding, padding), mode='replicate')

        # Use 'unfold' to efficiently create sliding windows without a for-loop.
        # windows shape: (B, T, window_size)
        windows = movement_padded.unfold(dimension=1, size=self.window_size, step=1)
        
        # Calculate the mean of each window to get the raw weights.
        # weights_raw shape: (B, T)
        weights_raw = windows.mean(dim=2)

        # 3. Normalize the weights using Softmax.
        # weights shape: (B, T)
        weights = F.softmax(weights_raw, dim=-1)
        
        # 4. Compute MSE loss and apply the weights.
        # Expand weights from (B, T) to (B, T, feat_dim) to match the MSE output shape.
        weights = weights.unsqueeze(-1).expand_as(gt_motion)

        # Calculate element-wise MSE loss by setting reduction='none'.
        mse_loss_none = F.mse_loss(pred_motion, gt_motion.detach(), reduction='none')
        
        # Apply the calculated weights to the MSE loss.
        weighted_loss = mse_loss_none * weights

        # 5. Aggregate the final loss value based on the specified reduction method.
        if self.reduction == 'mean':
            # Multiply by T to scale the loss, reflecting the intention of the paper
            # and the original implementation.
            return weighted_loss.mean() * T
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        elif self.reduction == 'none':
            return weighted_loss
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}")

if __name__ == '__main__':
    # Example code for testing
    batch_size = 4
    seq_len = 100
    features = 5023 * 3

    # Generate random motion data
    ground_truth = torch.randn(batch_size, seq_len, features)
    # Create predicted motion that is close to the ground truth
    predicted_motion = ground_truth + torch.randn(batch_size, seq_len, features) * 0.1

    # Instantiate the loss function
    loss_fn = PhoneticContextAwareLoss(window_size=5, reduction='mean')
    
    # Calculate the loss
    loss = loss_fn(predicted_motion, ground_truth)
    
    print(f"Phonetic Context-Aware Loss (window_size=5, reduction='mean'): {loss.item()}")

    # Test with reduction='sum'
    loss_fn_sum = PhoneticContextAwareLoss(window_size=5, reduction='sum')
    loss_sum = loss_fn_sum(predicted_motion, ground_truth)
    print(f"Phonetic Context-Aware Loss (window_size=5, reduction='sum'): {loss_sum.item()}")

