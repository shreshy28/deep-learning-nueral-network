import numpy as np
import pytest
from nn.layers import Dense, BatchNorm, Dropout
from nn.activations import ReLU, Sigmoid, Softmax
from nn.losses import CrossEntropy, MSE
from nn.model import Sequential
from nn.optimizers import SGD, Adam
from nn.training import train, evaluate


def finite_difference(array, objective, epsilon=1e-5):
    result = np.empty_like(array)
    for index in np.ndindex(array.shape):
        original = array[index]
        array[index] = original+epsilon
        plus = objective()
        array[index] = original-epsilon
        minus = objective()
        array[index] = original
        result[index] = (plus-minus)/(2*epsilon)
    return result


@pytest.mark.parametrize('kind', ['dense', 'batchnorm', 'relu', 'sigmoid', 'softmax'])
def test_layer_gradients(kind):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5, 3))
    layer = {'dense': lambda: Dense(3, 4, rng), 'batchnorm': lambda: BatchNorm(3),
             'relu': ReLU, 'sigmoid': Sigmoid, 'softmax': Softmax}[kind]()
    upstream = rng.normal(size=layer.forward(x).shape)
    actual = layer.backward(upstream).copy()
    gradients = {k: v.copy() for k, v in getattr(layer, 'grads', {}).items()}
    objective = lambda: np.sum(layer.forward(x)*upstream)
    np.testing.assert_allclose(actual, finite_difference(x, objective), atol=2e-7, rtol=2e-5)
    for name, value in getattr(layer, 'params', {}).items():
        np.testing.assert_allclose(gradients[name], finite_difference(value, objective), atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize('criterion,targets', [(CrossEntropy(), np.array([0, 2, 1, 0])),
                                             (MSE(), np.ones((4, 3)))])
def test_loss_gradients(criterion, targets):
    x = np.random.default_rng(8).normal(size=(4, 3))
    criterion.forward(x, targets)
    actual = criterion.backward()
    np.testing.assert_allclose(actual, finite_difference(x, lambda: criterion.forward(x, targets)), atol=1e-7)


def test_network_l2_gradient():
    rng = np.random.default_rng(5)
    model = Sequential([Dense(2, 3, rng), Sigmoid(), Dense(3, 2, rng)])
    x, y, loss = rng.normal(size=(4, 2)), np.array([0, 1, 0, 1]), CrossEntropy()
    loss.forward(model.forward(x, True), y)
    model.backward(loss.backward())
    for _, value, gradient in model.parameters(.03):
        numerical = finite_difference(value, lambda: loss.forward(model.forward(x, True), y)+model.penalty(.03))
        np.testing.assert_allclose(gradient, numerical, atol=1e-7)


def test_stability():
    x = np.array([[1e4, -1e4], [-1e4, 1e4]])
    loss = CrossEntropy()
    assert np.isfinite(loss.forward(x, np.array([1, 0])))
    assert np.isfinite(loss.backward()).all()
    assert np.isfinite(Sigmoid().forward(x)).all()
    np.testing.assert_allclose(Softmax().forward(x).sum(axis=1), 1)


def test_dropout():
    layer = Dropout(.25, np.random.default_rng(1))
    x = np.ones((1000, 100))
    out = layer.forward(x)
    assert abs(out.mean()-1) < .01
    np.testing.assert_array_equal(layer.backward(x), out)
    np.testing.assert_array_equal(layer.forward(x, False), x)
    with pytest.raises(ValueError):
        Dropout(1, np.random.default_rng())


def test_batchnorm_inference_and_checkpoint(tmp_path):
    rng = np.random.default_rng(1)
    model = Sequential([Dense(2, 3, rng), BatchNorm(3), ReLU()])
    x = rng.normal(size=(10, 2))
    model.forward(x, True)
    expected = model.forward(x)
    np.testing.assert_allclose(model.forward(x[:1]), expected[:1])
    model.save(tmp_path/'model.npz')
    model.layers[0].params['W'][:] = 0
    model.load(tmp_path/'model.npz')
    np.testing.assert_array_equal(model.forward(x), expected)
    with pytest.raises(ValueError):
        Sequential([Dense(2, 4, rng)]).load(tmp_path/'model.npz')


@pytest.mark.parametrize('optimizer', [SGD(.1), SGD(.1, .9), Adam(.1)])
def test_optimizers(optimizer):
    x = np.array([3.])
    for _ in range(200):
        optimizer.step([('x', x, x.copy())])
    assert abs(x[0]) < .001


def test_training_schedule_early_stop_restore():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(100, 2))
    y = (x[:, 0] > 0).astype(int)
    model = Sequential([Dense(2, 2, rng)])
    optimizer = Adam(.05)
    history = train(model, optimizer, x, y, x, y, epochs=20, batch_size=20, decay=.95)
    assert evaluate(model, x, y)[1] > .95
    assert history[-1]['lr'] < history[0]['lr']
    # Frozen parameters force a validation plateau and exercise early stopping.
    history = train(model, SGD(0), x, y, x, y, epochs=20, patience=2)
    assert len(history) == 3
    assert evaluate(model, x, y)[0] == pytest.approx(min(r['validation_loss'] for r in history))


def test_dropout_fixed_mask_gradient():
    rng = np.random.default_rng(2)
    layer = Dropout(.3, np.random.default_rng(10))
    x = rng.normal(size=(3, 4))
    upstream = rng.normal(size=x.shape)
    layer.forward(x)
    mask = layer.mask.copy()
    actual = layer.backward(upstream)
    np.testing.assert_allclose(actual, finite_difference(x, lambda: np.sum(x*mask*upstream)), atol=1e-9)


def test_data_synthetic_reproducibility():
    from nn.data import synthetic
    a, b = synthetic(42)
    c, d = synthetic(42)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, d)
    assert a.shape == (2400, 2)
    assert set(b) == {0, 1, 2}


def test_checkpoint_batchnorm_parameters_gradient():
    rng = np.random.default_rng(19)
    model = Sequential([Dense(2, 3, rng), BatchNorm(3), Sigmoid(), Dense(3, 2, rng)])
    x, y, loss = rng.normal(size=(5, 2)), np.array([0, 1, 0, 1, 1]), CrossEntropy()
    loss.forward(model.forward(x, True), y)
    model.backward(loss.backward())
    gradients = [(v, g.copy()) for _, v, g in model.parameters()]
    for value, gradient in gradients:
        np.testing.assert_allclose(gradient, finite_difference(value, lambda: loss.forward(model.forward(x, True), y)), atol=1e-7)
