import numpy as np
from .losses import CrossEntropy

def evaluate(model, x, y, batch_size=512):
    loss, correct = 0., 0
    for start in range(0, len(x), batch_size):
        xb, yb = x[start:start+batch_size], y[start:start+batch_size]
        logits = model.forward(xb)
        loss += len(xb)*CrossEntropy().forward(logits, yb)
        correct += np.sum(logits.argmax(axis=1) == yb)
    return float(loss/len(x)), float(correct/len(x))

def train(model, optimizer, x, y, xv, yv, epochs=30, batch_size=128,
          seed=42, l2=0., patience=8, decay=0.97):
    """Exponential LR schedule; restore best validation-loss parameters."""
    rng, history = np.random.default_rng(seed), []
    best, stale, best_state = np.inf, 0, model.state()
    criterion = CrossEntropy()
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        lr = optimizer.lr
        for start in range(0, len(x), batch_size):
            ids = order[start:start+batch_size]
            criterion.forward(model.forward(x[ids], training=True), y[ids])
            model.backward(criterion.backward())
            optimizer.step(model.parameters(l2))
        tl, ta = evaluate(model, x, y)
        vl, va = evaluate(model, xv, yv)
        row = dict(epoch=epoch+1, lr=lr, train_loss=tl, train_accuracy=ta,
                   validation_loss=vl, validation_accuracy=va, l2_penalty=float(model.penalty(l2)))
        history.append(row)
        print(f"epoch {epoch+1:02d}: loss {tl:.4f}/{vl:.4f}, accuracy {ta:.4f}/{va:.4f}", flush=True)
        if vl < best-1e-5:
            best, stale, best_state = vl, 0, model.state()
        else:
            stale += 1
        optimizer.lr *= decay
        if stale >= patience:
            break
    model.restore(best_state)
    return history
