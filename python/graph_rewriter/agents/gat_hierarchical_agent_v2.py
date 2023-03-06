import time
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import jraph
import optax
from optax import linear_schedule, scale_by_adam, scale_by_schedule
from flax.training.train_state import TrainState
from flax.training import checkpoints
from tensorflow_probability.substrates import jax as tfp

from graph_rewriter.agents.base_agent import (_BaseAgent, compute_gae,
                                              action_from_logits,
                                              masked_logits)
from graph_rewriter.agents.models import (MainTwinHeadModel_GAT_global_v2,
                                          SubTwinHeadModel_GAT_global_v2)


class GATHierarchicalPPOAgent_v2(_BaseAgent):
    """A HierarchicalPPOAgent groups two individuals ppo agents
    and manage state
    """

    def __init__(
        self,
        num_actions: int,  # num_action for main model
        num_candidates: int,  # num_action for sub model

        # gnn related
        num_head: int,
        hidden_dim: int,
        message_passing_steps: int,
        gat_global_update_mlp: int,

        # agent related
        key: PRNGKey,
        state_input: dict,  # for init model
        gamma_discount: float,
        gae_lambda: float,
        learning_rate: float,
        clip_ratio: float,
        global_norm: float,
        target_kl: Optional[float],
        policy_feature: list[int],
        vf_feature: list[int],
        n_minibatch: int,
        update_round: int,

        # utils
        name: str,  # name of the DNNs
        checkpoint_path: str,
        regular_checkpoint_dir: str,
        num_episodes: int,
        episodes_per_batch: int,
    ):

        super().__init__(name, checkpoint_path, regular_checkpoint_dir)
        assert (update_round > 0)

        # attr
        self.num_actions = num_actions
        self.num_candidates = num_candidates
        self.gamma_discount = gamma_discount
        self.gae_lambda = gae_lambda
        self.learning_rate = learning_rate
        self.clip_ratio = clip_ratio
        self.update_round = update_round
        self.n_minibatch = n_minibatch
        self.target_kl = target_kl
        self.update_step = 0
        self.key, k1, k2 = jax.random.split(key, 3)

        # build each components
        # main agent; action + 1 for no-op
        main_model = MainTwinHeadModel_GAT_global_v2(num_head,
                                                     hidden_dim,
                                                     message_passing_steps,
                                                     gat_global_update_mlp,
                                                     policy_feature,
                                                     vf_feature,
                                                     num_actions=num_actions +
                                                     1)

        # sub agent
        sub_model = SubTwinHeadModel_GAT_global_v2(num_head, hidden_dim,
                                                   message_passing_steps,
                                                   gat_global_update_mlp,
                                                   policy_feature, vf_feature)

        # opt
        schedule_fn = linear_schedule(
            init_value=-learning_rate,
            # end_value=-0.0005,
            end_value=0,
            transition_steps=num_episodes * update_round // episodes_per_batch)

        main_opt = optax.chain(optax.clip_by_global_norm(global_norm),
                               scale_by_adam(eps=1e-5),
                               scale_by_schedule(schedule_fn))

        sub_opt = optax.chain(optax.clip_by_global_norm(global_norm),
                              scale_by_adam(eps=1e-5),
                              scale_by_schedule(schedule_fn))

        # main_opt = optax.chain(optax.clip_by_global_norm(global_norm),
        #                        optax.adam(learning_rate, eps=1e-5))
        # sub_opt = optax.chain(optax.clip_by_global_norm(global_norm),
        #                       optax.adam(learning_rate, eps=1e-5))

        # build state
        main_param = main_model.init(k1, state_input["graph"], 1)

        # GraphsTuple is padded to num_candidates
        # so it is constant input shape
        # need to find the index, where xfers is not None
        n = len(state_input["mask"])
        for i in range(n):
            if state_input["mask"][i] != 0:
                break

        sub_param = sub_model.init(k2, state_input["xfers"][i], 1)

        self.main_state = TrainState.create(apply_fn=main_model.apply,
                                            params=main_param,
                                            tx=main_opt)

        self.sub_state = TrainState.create(apply_fn=sub_model.apply,
                                           params=sub_param,
                                           tx=sub_opt)

    def act(self, states: dict, explore: bool):
        assert (isinstance(states, dict)), "states type"
        self.key, k1, k2 = jax.random.split(self.key, 3)

        # ========= main action =========
        main_logits, main_vf_values = self.main_state.apply_fn(
            self.main_state.params, states["graph"], 1)
        # invalid masking
        main_logits = masked_logits(main_logits, states["mask"])
        # sample from logits
        main_action, main_logprobs = action_from_logits(
            main_logits, k1, explore)

        # build sub-state from main action
        graph_tuples, masks = _build_sub_state_from_main_action(
            states, main_action)

        # ========= sub action =========
        sub_logits, sub_vf_values = self.sub_state.apply_fn(
            self.sub_state.params, graph_tuples, 1)
        # invalid masking
        sub_logits = masked_logits(sub_logits, masks)
        # sample from logits
        sub_action, sub_logprobs = action_from_logits(sub_logits, k2, explore)

        # each term shape: [B, ]; for single environment -> (1, )
        return main_action, main_logprobs, main_vf_values, sub_action, sub_logprobs, sub_vf_values

    def update(self, states: list[dict], main_actions: list[jnp.ndarray],
               main_log_probs: list[jnp.ndarray],
               main_vf_values: list[jnp.ndarray],
               sub_actions: list[jnp.ndarray],
               sub_log_probs: list[jnp.ndarray],
               sub_vf_values: list[jnp.ndarray], rewards: list[float],
               dones: list[bool]) -> dict[str, float]:
        """
        Computes proximal policy updates and value function updates
        """
        n_rollout = len(states)

        # convert to jnp
        main_actions = jnp.array(main_actions,
                                 dtype=jnp.int32)  # use for update
        sub_actions = jnp.array(sub_actions, dtype=jnp.int32)  # use for update
        dones = jnp.array(dones, dtype=jnp.int32)  # use for gae

        main_log_probs = jnp.array(main_log_probs, dtype=jnp.float32)
        sub_log_probs = jnp.array(sub_log_probs, dtype=jnp.float32)
        main_vf_values = jnp.array(main_vf_values, dtype=jnp.float32)
        sub_vf_values = jnp.array(sub_vf_values, dtype=jnp.float32)
        rewards = jnp.array(rewards, dtype=jnp.float32)

        # convert to shape [n_rollout, ], because e.g. main_actions dim is 1
        # Generally can be [n_rollout, num_feature]
        main_actions = jnp.reshape(main_actions, (n_rollout, ))
        sub_actions = jnp.reshape(sub_actions, (n_rollout, ))
        dones = jnp.reshape(dones, (n_rollout, ))
        main_log_probs = jnp.reshape(main_log_probs, (n_rollout, ))
        sub_log_probs = jnp.reshape(sub_log_probs, (n_rollout, ))
        main_vf_values = jnp.reshape(main_vf_values, (n_rollout, ))
        sub_vf_values = jnp.reshape(sub_vf_values, (n_rollout, ))
        rewards = jnp.reshape(rewards, (n_rollout, ))
        # GAE
        main_gae, main_returns = compute_gae(main_vf_values, rewards, dones,
                                             self.gamma_discount,
                                             self.gae_lambda)
        sub_gae, sub_returns = compute_gae(sub_vf_values, rewards, dones,
                                           self.gamma_discount,
                                           self.gae_lambda)
        # gae is one element less so reduced trajectory
        # otherwise we need to bootstrap
        # main_actions = main_actions[:-1]
        # sub_actions = sub_actions[:-1]
        # dones = dones[:-1]
        # main_log_probs = main_log_probs[:-1]
        # sub_log_probs = sub_log_probs[:-1]
        # main_vf_values = main_vf_values[:-1]
        # sub_vf_values = sub_vf_values[:-1]
        # rewards = rewards[:-1]
        # prepare state input
        main_graph_list, sub_graph_list, main_mask, sub_mask = _build_hierarchical_states(
            states, main_actions, 0)

        # multiple update rounds
        info = {
            "main_actor_loss": 0.,
            "main_vf_loss": 0.,
            "main_entropy": 0.,
            "main_kl": 0.,
            "sub_actor_loss": 0.,
            "sub_vf_loss": 0.,
            "sub_entropy": 0.,
            "sub_kl": 0.,
        }
        total_train_size = len(main_actions)
        idxes = jnp.arange(total_train_size)
        # steps_per_epoch = total_train_size // self.mini_batch_size
        mini_batch_size = total_train_size // self.n_minibatch
        if mini_batch_size == 0:
            mini_batch_size = 1
        cnt = 0
        for i in range(self.update_round):
            # build random indexes
            self.key, k = jax.random.split(self.key, 2)
            idxes = jax.random.permutation(k, idxes)
            # NOTE: skip incomplete batch, so shape are constant and jittable
            # idxes = idxes[:steps_per_epoch * mini_batch_size]
            # idxes = idxes.reshape((steps_per_epoch, mini_batch_size))
            idxes_list = [
                idxes[start:start + mini_batch_size]
                for start in jnp.arange(0, total_train_size, mini_batch_size)
            ]

            # mini-batch for update
            start_time = time.perf_counter()
            for j, idx in enumerate(idxes_list):
                self.main_state, main_actor_loss, main_vf_loss, main_entropy, main_approx_kl, main_ratio = _update(
                    False, self.main_state, jnp.take(main_gae, idx),
                    jnp.take(main_returns, idx), jnp.take(main_actions, idx),
                    jnp.take(main_log_probs,
                             idx), jnp.take(main_vf_values, idx),
                    [main_graph_list[ii] for ii in idx],
                    jnp.take(main_mask, idx, axis=0), self.clip_ratio)
                self.sub_state, sub_actor_loss, sub_vf_loss, sub_entropy, sub_approx_kl, sub_ratio = _update(
                    False, self.sub_state, jnp.take(sub_gae, idx),
                    jnp.take(sub_returns, idx), jnp.take(sub_actions, idx),
                    jnp.take(sub_log_probs, idx), jnp.take(sub_vf_values, idx),
                    [sub_graph_list[ii] for ii in idx],
                    jnp.take(sub_mask, idx, axis=0), self.clip_ratio)

                # DEBUG:
                if i == 0 and j == 0:
                    print(f"[update] round {i}: {mini_batch_size}")
                    print(main_ratio)
                    print(sub_ratio)
                elif j == 0:
                    print(f"[update] round {i}: {mini_batch_size}")
                    print(main_ratio)
                    print(sub_ratio)

                # per-minibatch STATS
                cnt += 1
                info["main_actor_loss"] += main_actor_loss
                info["main_vf_loss"] += main_vf_loss
                info["main_entropy"] += main_entropy
                info["main_kl"] += main_approx_kl
                info["sub_actor_loss"] += sub_actor_loss
                info["sub_vf_loss"] += sub_vf_loss
                info["sub_entropy"] += sub_entropy
                info["sub_kl"] += sub_approx_kl

            if self.target_kl is not None and self.update_step > 10:
                if main_approx_kl > self.target_kl or sub_approx_kl > self.target_kl:
                    print(
                        f"early stopping at round {i+1} out of {self.update_round} - main KL: {main_approx_kl:.4f} - sub KL: {sub_approx_kl:.4f} - target KL: {self.target_kl:.4f}"
                    )
                    break

        t = time.perf_counter() - start_time
        print(f"[update] took {t:.2f} seconds")
        self.update_step += 1
        for key, v in info.items():
            info[key] = v / cnt
        return info

    def regular_save(self, step, best_inference_runtime):
        """
        regularly save agent state (possibly for training multiple graphs)
            (even Kerberos ticket expires, it can still save... why?)
        """
        meta = {
            "episode": step,
            "update_step": self.update_step,
            "key": self.key,
            "best_inference_runtime": best_inference_runtime
        }
        checkpoints.save_checkpoint(ckpt_dir=self.regular_checkpoint_dir,
                                    target=meta,
                                    step=step,
                                    prefix="checkpoint_meta_",
                                    keep=5)
        checkpoints.save_checkpoint(ckpt_dir=self.regular_checkpoint_dir,
                                    target=self.main_state,
                                    step=step,
                                    prefix="checkpoint_main_",
                                    keep=5)
        checkpoints.save_checkpoint(ckpt_dir=self.regular_checkpoint_dir,
                                    target=self.sub_state,
                                    step=step,
                                    prefix="checkpoint_sub_",
                                    keep=5)

    def save(self, step, best_inference_runtime):
        meta = {
            "episode": step,
            "key": self.key,
            "update_step": self.update_step,
            "best_inference_runtime": best_inference_runtime
        }
        checkpoints.save_checkpoint(ckpt_dir=self.checkpoint_root,
                                    target=meta,
                                    step=step,
                                    prefix="checkpoint_meta_",
                                    keep=5)
        checkpoints.save_checkpoint(ckpt_dir=self.checkpoint_root,
                                    target=self.main_state,
                                    step=step,
                                    prefix="checkpoint_main_",
                                    keep=5)
        checkpoints.save_checkpoint(ckpt_dir=self.checkpoint_root,
                                    target=self.sub_state,
                                    step=step,
                                    prefix="checkpoint_sub_",
                                    keep=5)

    def load_regular(self, step=None) -> dict:
        meta = {
            "episode": 0,
            "update_step": 0,
            "best_inference_runtime": float("inf"),
            "key": PRNGKey(0)
        }
        restored_meta = checkpoints.restore_checkpoint(
            ckpt_dir=self.regular_checkpoint_dir,
            target=meta,
            step=step,
            prefix="checkpoint_meta_")
        restored_main = checkpoints.restore_checkpoint(
            ckpt_dir=self.regular_checkpoint_dir,
            target=self.main_state,
            step=step,
            prefix="checkpoint_main_")
        restored_sub = checkpoints.restore_checkpoint(
            ckpt_dir=self.regular_checkpoint_dir,
            target=self.sub_state,
            step=step,
            prefix="checkpoint_sub_")
        # directly assign
        self.main_state = restored_main
        self.sub_state = restored_sub
        self.key = restored_meta["key"]
        self.update_step = restored_meta["update_step"]
        return restored_meta

    def load(self, step=None) -> dict:
        """
        Args:
            step: if None, get the latest checkpoint
        """
        meta = {
            "episode": 0,
            "update_step": 0,
            "best_inference_runtime": float("inf"),
            "key": PRNGKey(0)
        }
        restored_meta = checkpoints.restore_checkpoint(
            ckpt_dir=self.checkpoint_root,
            target=meta,
            step=step,
            prefix="checkpoint_meta_")
        restored_main = checkpoints.restore_checkpoint(
            ckpt_dir=self.checkpoint_root,
            target=self.main_state,
            step=step,
            prefix="checkpoint_main_")
        restored_sub = checkpoints.restore_checkpoint(
            ckpt_dir=self.checkpoint_root,
            target=self.sub_state,
            step=step,
            prefix="checkpoint_sub_")
        # directly assign
        self.main_state = restored_main
        self.sub_state = restored_sub
        self.key = restored_meta["key"]
        self.update_step = restored_meta["update_step"]

        # FOR save and store immediately checking only
        # assert jax.tree_util.tree_all(
        #     jax.tree_map(lambda x, y: (x == y).all(), self.main_state.params,
        #                  restored_main.params))
        # assert jax.tree_util.tree_all(
        #     jax.tree_map(lambda x, y: (x == y).all(), self.sub_state.params,
        #                  restored_sub.params))
        return restored_meta


def _update(jit: bool, train_state: TrainState, gae, returns, old_actions,
            old_log_probs, old_vf, old_graph_input, mask, clip_eps):
    # convert list -> graph batch; IMPLICITLY BATCH
    graph = jraph.batch(old_graph_input)
    batch_size = len(old_graph_input)
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)

    # NOTE: if want to JIT, the graph here should be padded
    # but then we cannot control the batch size for
    # flattening the global features
    # so I dont think we can JIT in this kinds of training
    if jit:
        return _update_jit(train_state, gae, returns, old_actions,
                           old_log_probs, old_vf, graph, mask, clip_eps)
    else:
        return _update_not_jit(train_state, gae, returns, old_actions,
                               old_log_probs, old_vf, graph, mask, batch_size,
                               clip_eps)


def _update_not_jit(train_state: TrainState, gae, returns, old_actions,
                    old_log_probs, old_vf, graph: jraph.GraphsTuple, mask,
                    batch_size, clip_eps):

    def loss(params, fn, gae, returns, old_actions, old_log_probs, old_vf,
             graph, mask):
        new_logits, new_vf = fn(params, graph, batch_size)
        new_logits = masked_logits(new_logits, mask)

        # new dist from: new logits, old action
        dist = tfp.distributions.Categorical(new_logits)
        new_log_prob, entropy = dist.log_prob(old_actions), dist.entropy()

        # ENTROPY
        entropy_bonus = jnp.mean(entropy)

        # VF (critic) also clip
        value_loss1 = (new_vf - returns)**2
        vf_clip = old_vf + (new_vf - old_vf).clip(-clip_eps, clip_eps)
        value_loss2 = (vf_clip - returns)**2
        vf_loss = 0.5 * jnp.mean(jnp.maximum(value_loss1, value_loss2))

        # ACTOR CLIP
        log_ratio = new_log_prob - old_log_probs
        ratio = jnp.exp(new_log_prob - old_log_probs)
        # gae is normalised
        trpo_gain = ratio * gae
        clip_gain = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae
        min_gain = jnp.minimum(trpo_gain, clip_gain)
        actor_loss = -jnp.mean(min_gain)

        # JOINT
        J = actor_loss + 0.5 * vf_loss - 0.01 * entropy_bonus

        # Approx KL
        approx_kl = jnp.mean((ratio - 1) - log_ratio)
        return J, (actor_loss, vf_loss, entropy_bonus, approx_kl, ratio)

    # loss and update
    grad_fn = jax.value_and_grad(loss, has_aux=True)
    all_losses, grads = grad_fn(train_state.params, train_state.apply_fn, gae,
                                returns, old_actions, old_log_probs, old_vf,
                                graph, mask)
    train_state = train_state.apply_gradients(grads=grads)
    # unpack
    J, (actor_loss, vf_loss, entropy_bonus, approx_kl, ratio) = all_losses
    return train_state, actor_loss, vf_loss, entropy_bonus, approx_kl, ratio


@partial(jax.jit, static_argnames=("clip_eps"))
def _update_jit(train_state: TrainState, gae, returns, old_actions,
                old_log_probs, old_vf, graph: jraph.GraphsTuple, mask,
                clip_eps):
    """a way to debug is to print inside JIT function
    because each compilation will print once
    """
    print("calling jit", end="; ")
    return None


def _build_hierarchical_states(states: list[dict], main_actions: jnp.ndarray,
                               gae_reduce: int) -> tuple:
    """
    Args:
        gae_reduce: account for gae needs 1 element less
    """
    n_rollout = len(states) - gae_reduce  # account for reduction

    # prepare masks
    main_mask = []
    sub_mask = []
    for i in range(n_rollout):
        arr = states[i]["mask"]
        arr = jnp.reshape(arr, (1, -1))
        main_mask.append(arr)

        xfer_id = int(main_actions[i])
        sub_arr = states[i]["candidates_mask"][xfer_id]
        sub_arr = jnp.reshape(sub_arr, (1, -1))
        sub_mask.append(sub_arr)

    # shape: [n_rollout, num_block];
    main_mask = jnp.concatenate(main_mask, axis=0)
    # shape: [n_rollout, num_candidates];
    # sub_mask can be stacked, because it is padded to max_num_candidates
    sub_mask = jnp.concatenate(sub_mask, axis=0)

    # prepare graph input
    # Graph shape: [n_rollout, ];
    main_graph_input = []
    # Graph shape: [n_rollout, max_num_candidates];
    sub_graph_input = []
    for i in range(n_rollout):
        main_g = states[i]["graph"]
        xfer_id = int(main_actions[i])
        g = states[i]["xfers"][xfer_id]
        main_graph_input.append(main_g)
        sub_graph_input.append(g)
    # if batch, has n_rollout GraphsTuple; each graph has max_num_candidates

    return main_graph_input, sub_graph_input, main_mask, sub_mask


def _build_sub_state_from_main_action(states, main_action) -> tuple:
    xfer_id = int(main_action)
    return states["xfers"][xfer_id], states["candidates_mask"][xfer_id]
