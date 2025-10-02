import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


# -------- 1. define a function to optimize (a "wobbly" 1D function) ----------
def f(x):
    return jnp.sin(3.0 * x) * jnp.exp(-0.1 * x**2)


# -------- 2. compute gradient automatically ----------------------------------
grad_f = jax.grad(f)

# pick a starting point
x = 2.5

xs = [x]
for step in range(25):
    g = grad_f(x)
    x = x - 0.2 * g  # gradient descent
    xs.append(x)

print("Final x ≈", x, "f(x) ≈", f(x))

# -------- 3. visualize the path ----------------------------------------------
xs_plot = jnp.linspace(-4, 4, 400)
ys_plot = f(xs_plot)

plt.figure(figsize=(6, 4))
plt.plot(xs_plot, ys_plot, label="f(x)")
plt.scatter(xs, f(jnp.array(xs)), color="red", s=30, label="gradient descent path")
plt.legend()
plt.title("JAX Gradient Descent Demo")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()
