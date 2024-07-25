import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from PIL import Image

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import hydra

from safetensors.flax import load_file
from flax.traverse_util import unflatten_dict

from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper, CTRolloutManager


class ScannedRNN(nn.Module):

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        hidden_size = ins.shape[-1]
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(hidden_size, *ins.shape[:-1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )


class RNNQNetwork(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int
    hidden_dim: int
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, obs, dones):
        embedding = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        q_vals = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(embedding)

        return hidden, q_vals



@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):

    file_path = '/home/stefan/Code/JaxMARLFork/JaxMARL/models/coin_game/iql_rnn_coin_game_seed0_vmap0.safetensors'
    key = jax.random.PRNGKey(0)
    key, key_reset = jax.random.split(key, 2)

    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.agents], axis=0)

    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(env.agents)}
    
    # INIT ENV
    env = make('coin_game')
    env = LogWrapper(env)
    wrapped_env = CTRolloutManager(env, batch_size=1)

    network = RNNQNetwork(
            action_dim=wrapped_env.max_action_space,
            hidden_dim=64,
        )
    
    hstate = ScannedRNN.initialize_carry(64, len(env.agents), 1)

    obs, log_env_state = wrapped_env.batch_reset(key_reset)
    dones = {agent:jnp.zeros((1), dtype=bool) for agent in env.agents+['__all__']}

    flattened_params = load_file(file_path)
    params = unflatten_dict(flattened_params, sep=',')

    param_dims = jax.tree_util.tree_map(lambda x: x.shape, params)
    #print(param_dims)

    rng, key_s = jax.random.split(jax.random.PRNGKey(0), 2)
    images = []
    for _ in range(100):
        
        images.append(env.render(log_env_state.env_state))

        #print(f'obs: {obs}')
        #print(f'dones: {dones}')

        _obs = batchify(obs)[:, np.newaxis]
        _dones = batchify(dones)[:, np.newaxis]

        #print(f'batchified_obs: {_obs}')
        #print(f'batchified_dones: {_dones}')

        #hstate, q_vals = homogeneous_pass(params, hstate, obs_, dones_)

        hstate, q_vals = jax.vmap(
                    network.apply, in_axes=(None, 0, 0, 0)
                )(  # vmap across the agent dim
                    params,
                    hstate,
                    _obs,
                    _dones,
                )

        print({agent: jnp.argmax(jnp.squeeze(val)) for agent, val in enumerate(q_vals)})

        actions = {agent: jnp.asarray([jnp.argmax(jnp.squeeze(val))]) for agent, val in enumerate(q_vals)}

        obs, log_env_state, _, dones, _ = wrapped_env.batch_step(key_s, log_env_state, actions)

        rng, key_s = jax.random.split(rng, 2)


    gif = [i.convert("P", palette=Image.ADAPTIVE) for i in images]
    gif[0].save('prediction_shared.gif', save_all=True, optimize=False, append_images=gif[1:], loop=0)

if __name__ == "__main__":
    main()
