import torch
import torch.nn.functional as F
import math

def pad_time_to_multiple(x: torch.Tensor, multiple: int):
    """
    Pads the time dimension (dim=1) to the next multiple.
    x: [B, T, F]
    """
    T = x.shape[1]
    T_pad = 1024#int(math.ceil(T / multiple) * multiple)
    pad_right = T_pad - T
    if pad_right > 0:
        # Pad time dimension on the right
        x = F.pad(x, (0, 0, 0, pad_right))  # (F_left, F_right, T_left, T_right)
    return x, pad_right


def collate_fn(batch, unet_multiple: int = 18):
    """
    Output shape per tensor:
      [batch_size, 1, padded_num_steps, feature_dim]

    unet_multiple:
      Should be >= 2^(#down_blocks). For your UNet, 16 or 32 is safe.
    """

    def pad_and_unsqueeze(seq_list):
        # Step 1: pad variable-length episodes → [B, T_max, F]
        x = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(s, dtype=torch.float32) for s in seq_list],
            batch_first=True
        )

        # Step 2: pad time dimension to UNet-safe multiple
        x, pad_right = pad_time_to_multiple(x, unet_multiple)

        # Step 3: add channel dim → [B, 1, T_pad, F]
        return x, pad_right

    observations, pad_obs = pad_and_unsqueeze([x.observations for x in batch])
    actions,      pad_act = pad_and_unsqueeze([x.actions for x in batch])
    rewards,      pad_rew = pad_and_unsqueeze([x.rewards for x in batch])
    terminations, pad_ter = pad_and_unsqueeze([x.terminations for x in batch])
    truncations,  pad_tru = pad_and_unsqueeze([x.truncations for x in batch])

    # All pad_right values should be identical
    pad_right = pad_obs

    return {
        "id": torch.tensor([x.id for x in batch]),
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminations": terminations,
        "truncations": truncations,
        "pad_right": pad_right,  # <-- IMPORTANT for cropping after UNet
    }


## since each episode can be of varying length you must pad the squences
# def collate_fn(batch):
#     """
#     Collate function to combine data in a specified order.
#     Each sample in unsqueezed to add an extra dimension to train a UNet 1D Model
#     Output shape: [batch_size, 1, num_steps_in_episode, size_of(observations, rewards, actions etc.)]
#     """
#     return {
#         "id": torch.Tensor([x.id for x in batch]),
#         "observations": torch.nn.utils.rnn.pad_sequence(
#             [torch.as_tensor(x.observations, dtype=torch.float32) for x in batch],
#             batch_first=True
#         ).unsqueeze(1),
#         "actions": torch.nn.utils.rnn.pad_sequence(
#             [torch.as_tensor(x.actions, dtype=torch.float32) for x in batch],
#             batch_first=True
#         ).unsqueeze(1),
#         "rewards": torch.nn.utils.rnn.pad_sequence(
#             [torch.as_tensor(x.rewards, dtype=torch.float32) for x in batch],
#             batch_first=True,
#         ).unsqueeze(1),
#         "terminations": torch.nn.utils.rnn.pad_sequence(
#             [torch.as_tensor(x.terminations, dtype=torch.float32) for x in batch],
#             batch_first=True
#         ).unsqueeze(1),
#         "truncations": torch.nn.utils.rnn.pad_sequence(
#             [torch.as_tensor(x.truncations, dtype=torch.float32) for x in batch],
#             batch_first=True
#         ).unsqueeze(1),
#     }