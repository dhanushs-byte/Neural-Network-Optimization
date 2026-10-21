import matplotlib.pyplot as plt

x, target = 2.0, 10.0
def loss(w, b): return (w*x + b - target)**2
def gradients(w, b):
    error = w*x + b - target
    return 2*error*x, 2*error

w, b = 1.0, 0.0
dw, db = gradients(w, b)
print("Initial loss:", loss(w, b))
print("Hand gradient: dL/dw =", dw, ", dL/db =", db)

lr, losses = 0.05, []
for step in range(1, 51):
    dw, db = gradients(w, b)
    w -= lr*dw
    b -= lr*db
    losses.append(loss(w, b))
    if step % 10 == 0:
        print(f"Step {step}: loss = {losses[-1]:.8f}")
print("Final w:", w, "Final b:", b)

plt.plot(range(1, 51), losses)
plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Gradient Descent (lr=0.05)")
plt.grid(True); plt.tight_layout(); plt.savefig("task3_loss.png"); plt.show()

plt.figure()
for rate in [0.001, 0.05, 0.5]:
    w, b, vals = 1.0, 0.0, []
    for _ in range(50):
        dw, db = gradients(w, b)
        w -= rate*dw; b -= rate*db
        vals.append(loss(w, b))
    plt.plot(range(1, 51), vals, label=f"lr={rate}")
plt.yscale("log"); plt.xlabel("Step"); plt.ylabel("Loss (log scale)")
plt.title("Learning Rate Comparison"); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig("task3_learning_rate_comparison.png"); plt.show()
