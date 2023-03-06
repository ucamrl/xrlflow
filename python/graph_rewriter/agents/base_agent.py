from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from tensorflow_probability.substrates import jax as tfp


class _BaseAgent(ABC):
    """
    provide agent-level utility
    """

    def __init__(self, name, checkpoint_path, regular_checkpoint_dir):
        self.name = name
        self.checkpoint_root = checkpoint_path
        self.regular_checkpoint_dir = regular_checkpoint_dir

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def update(self):
        pass


def action_from_logits(logits: jnp.ndarray, key: PRNGKey, explore: bool):
    # logits shape: [B, num_actions]; action shape (B, )
    # if input raw logits, it will apply softmax to get the final prob
    dist = tfp.distributions.Categorical(logits=logits)
    if explore:
        action = dist.sample(seed=key)
    else:
        action = jnp.argmax(logits, axis=-1)

    # hand-roll Categorical log prob
    # shape: [B, num_actions]
    # log_probs = jax.nn.log_softmax(logits)
    # # shape: [B, num_actions]
    # one_hot = jax.nn.one_hot(jnp.squeeze(action),
    #                          num_classes=self.num_actions)
    # # shape: [B, ]; sum is because only one term is non-zero
    # action_log_prob = jnp.sum(one_hot * log_probs, axis=1)
    action_log_prob = dist.log_prob(action)
    return action, action_log_prob


def masked_logits(logits, mask):
    """
    Masks out invalid actions.

    Args:
        logits: Action logits, shape [B, num_actions]
        mask: Valid actions, shape [B, num_actions]

    Returns:
        masked_logits: tensor of shape [B, num_actions] where all values which were 0 in the corresponding
            mask are now -1e10. This means they will not be sampled when sampling actions from the policy.
    """
    mask_value = jnp.full(shape=logits.shape,
                          fill_value=-1e10,
                          dtype=logits.dtype)
    applied = jnp.where(mask, logits, mask_value)  # mask is 0 or 1 array
    return applied


def compute_gae(vf, reward, dones, gamma, gae_lambda):
    """
    gae compute cause one element less
        this can be address by bootstrapping,
        but bootstrapping seems too clumsy
    Returns:
        gae and returns
    """
    advantages = []
    gae = 0.

    # taso specific
    assert (int(dones[-1]) == 1), "episode must finish"
    n = len(reward)

    for t in reversed(range(n)):
        if t == n - 1:
            value_diff = 0
        else:
            value_diff = gamma * vf[t + 1] * (1 - dones[t]) - vf[t]
        delta = reward[t] + value_diff  # td-estimate for advantages
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.append(gae)
    advantages = advantages[::-1]  # reverse for the correct order
    advantages = jnp.array(advantages, dtype=jnp.float32)

    # use un-normalised gae for returns; returns -> r_t + V(s_{t+1})
    # returns = advantages + vf[:-1]
    returns = advantages + vf
    return advantages, returns
