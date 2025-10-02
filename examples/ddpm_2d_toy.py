import math
import time
import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
import numpy as np
import optax
import matplotlib.pyplot as plt


# ---------------------------
# Utilities
# ---------------------------


def set_printopts():
    np.set_printoptions(precision=4, suppress=True)
    jnp.set_printoptions(precision=4)


def sinusoidal_time_embed(t, dim=32):
    """
    t: shape [B], values in [0,1]
    returns: [B, dim]
    """
    half = dim // 2
    freqs = jnp.exp(jnp.linspace(0.0, math.log(1000.0), half))
    angles = t[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


def mlp_init(rng, sizes):
    params = []
    for i in range(len(sizes) - 1):
        rng, wkey, bkey = random.split(rng, 3)
        fan_in = sizes[i]
        w = random.normal(wkey, (fan_in, sizes[i + 1])) / jnp.sqrt(fan_in)
        b = jnp.zeros((sizes[i + 1],))
        params.append((w, b))
    return rng, params


def mlp_apply(params, x, act=jax.nn.silu):
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = act(x)
    return x


# ---------------------------
# Data: Swiss-roll (2D)
# ---------------------------


def make_swiss_roll(n, rng_key, noise=0.1):
    """
    Classic 2D swiss roll paramization.
    """
    t = random.uniform(rng_key, (n,)) * 3.0 * jnp.pi + jnp.pi
    x = t * jnp.cos(t)
    y = t * jnp.sin(t)
    pts = jnp.stack([x, y], axis=1)
    pts = pts / 10.0  # shrink a bit
    pts = pts + noise * random.normal(rng_key, pts.shape)
    return pts


# ---------------------------
# Diffusion schedules
# ---------------------------


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02


def make_linear_schedule(T, beta_start, beta_end):
    betas = jnp.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bars = jnp.cumprod(alphas)
    return betas, alphas, alpha_bars


# q(x_t | x_0)
def q_sample(rng, x0, t_idx, alpha_bars):
    """
    x0: [B,2]
    t_idx: [B] int32 in [0, T-1]
    alpha_bars: [T]
    """
    a_bar = alpha_bars[t_idx][:, None]
    eps = random.normal(rng, x0.shape)
    xt = jnp.sqrt(a_bar) * x0 + jnp.sqrt(1.0 - a_bar) * eps
    return xt, eps


# ---------------------------
# Model: epsilon-theta(x_t, t)
# ---------------------------


class EpsModel:
    def __init__(self, rng, hidden=128, time_dim=32):
        self.time_dim = time_dim
        # input is 2D + time embedding
        sizes = [2 + time_dim, hidden, hidden, 2]
        rng, params = mlp_init(rng, sizes)
        self.params = params

    def __call__(self, params, x, t01):
        # x: [B,2], t01: [B] in [0,1]
        temb = sinusoidal_time_embed(t01, dim=self.time_dim)
        inp = jnp.concatenate([x, temb], axis=1)
        return mlp_apply(params, inp, act=jax.nn.silu)


# ---------------------------
# Training step
# ---------------------------


@dataclass
class TrainConfig:
    batch_size: int = 1024
    steps: int = 4000
    lr: float = 1e-3
    log_every: int = 200
    sample_every: int = 1000


def make_dataset(rng, n=20000):
    (dkey,) = random.split(rng, 1)
    data = make_swiss_roll(n, dkey, noise=0.08)
    return data


def sample_batch(rng, data, bs):
    idx = random.randint(rng, (bs,), 0, data.shape[0])
    return data[idx]


def train(rng, data, model, diff_cfg: DiffusionConfig, train_cfg: TrainConfig):
    T = diff_cfg.timesteps
    betas, alphas, a_bars = make_linear_schedule(
        T, diff_cfg.beta_start, diff_cfg.beta_end
    )

    opt = optax.adamw(train_cfg.lr)
    opt_state = opt.init(model.params)

    @jit
    def loss_and_grad(params, rng, batch):
        # choose random t for each sample
        t_idx = random.randint(rng, (batch.shape[0],), minval=0, maxval=T)
        t01 = (t_idx.astype(jnp.float32) + 0.5) / T
        x_t, eps = q_sample(rng, batch, t_idx, a_bars)
        pred = model(params, x_t, t01)
        # simple L2 between predicted noise and true noise
        loss = jnp.mean((pred - eps) ** 2)
        return loss

    @jit
    def train_step(params, opt_state, rng, batch):
        l, grads = value_and_grad(loss_and_grad)(params, rng, batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, l

    losses = []
    p = model.params
    gkey = rng
    for step in range(1, train_cfg.steps + 1):
        gkey, bkey, tkey = random.split(gkey, 3)
        batch = sample_batch(bkey, data, train_cfg.batch_size)
        p, opt_state, l = train_step(p, opt_state, tkey, batch)
        losses.append(float(l))
        if step % train_cfg.log_every == 0:
            print(f"[{step:5d}/{train_cfg.steps}] loss={l:.5f}")
    return p, losses, (betas, alphas, a_bars)


# ---------------------------
# Sampling (reverse diffusion)
# ---------------------------


def p_sample_loop(rng, params, model, sched, n_samples=2048):
    betas, alphas, a_bars = sched
    T = betas.shape[0]
    gkey = rng
    x = random.normal(gkey, (n_samples, 2))

    # Precompute
    alpha = alphas
    beta = betas
    alpha_bar = a_bars

    @functools.partial(jit, static_argnums=0)
    def one_step(t, x, params):
        """
        t: scalar timestep index (int)
        x: [B,2] current sample
        """
        t01 = (t + 0.5) / T
        t01 = jnp.full((x.shape[0],), t01, dtype=jnp.float32)
        eps_theta = model(params, x, t01)

        a_t = alpha[t]
        a_bar_t = alpha_bar[t]
        # eqn for predicting x_{t-1}
        mean = (1.0 / jnp.sqrt(a_t)) * (
            x - (beta[t] / jnp.sqrt(1.0 - a_bar_t)) * eps_theta
        )
        # add noise except for t=0
        noise = random.normal(random.PRNGKey(t), x.shape)
        x_prev = jnp.where(t > 0, mean + jnp.sqrt(beta[t]) * noise, mean)
        return x_prev

    # reverse loop
    for t in range(T - 1, -1, -1):
        x = one_step(t, x, params)
    return x


# ---------------------------
# Main
# ---------------------------


def plot_results(data, samples, savepath=None):
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.scatter(np.asarray(data[:, 0]), np.asarray(data[:, 1]), s=3, alpha=0.6)
    ax1.set_title("Real data (Swiss-roll)")
    ax1.set_aspect("equal")
    ax1.grid(True, linewidth=0.2)

    ax2.scatter(np.asarray(samples[:, 0]), np.asarray(samples[:, 1]), s=3, alpha=0.6)
    ax2.set_title("DDPM Samples")
    ax2.set_aspect("equal")
    ax2.grid(True, linewidth=0.2)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=160)
        print(f"Saved: {savepath}")
    plt.show()


def main():
    set_printopts()
    seed = 42
    rng = random.PRNGKey(seed)

    # Configs
    diff_cfg = DiffusionConfig(timesteps=500, beta_start=1e-4, beta_end=0.02)
    train_cfg = TrainConfig(batch_size=1024, steps=3000, lr=2e-3, log_every=200)

    print("Generating dataset...")
    rng, dkey = random.split(rng)
    data = make_dataset(dkey, n=20000)

    print("Building model...")
    rng, mkey = random.split(rng)
    model = EpsModel(mkey, hidden=128, time_dim=32)

    print("Training...")
    t0 = time.time()
    params, losses, sched = train(rng, data, model, diff_cfg, train_cfg)
    print(f"Training done in {time.time() - t0:.1f}s. Final loss={losses[-1]:.5f}")

    print("Sampling...")
    rng, skey = random.split(rng)
    samples = p_sample_loop(skey, params, model, sched, n_samples=4000)

    print("Plotting...")
    plot_results(data[:4000], samples, savepath=None)


if __name__ == "__main__":
    main()
