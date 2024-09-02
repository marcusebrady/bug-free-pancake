import numpy as np
from autograd import Tensor
from layers import ContinuousFilterConv, MessageBlock, UpdateBlock, Linear, Sequential, shifted_softplus
from functools import partial
import cProfile
import pstats
import io
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def profile(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        logger.debug(f"Profile for {func.__name__}:\n{s.getvalue()}")
        return result
    return wrapper

@profile
def test_continuous_filter_conv():
    logger.info("Testing ContinuousFilterConv")
    num_features = 64
    num_rbf = 20
    cutoff = 5.0
    cfconv = ContinuousFilterConv(num_features, num_rbf, cutoff)

    x = Tensor(np.random.randn(10, num_features))
    edge_index = Tensor(np.array([[0, 1, 2], [1, 2, 0]]))
    edge_attr = Tensor(np.random.rand(3, 1))

    logger.debug(f"Input shapes: x={x.shape}, edge_index={edge_index.shape}, edge_attr={edge_attr.shape}")
    output = cfconv(x, edge_index, edge_attr)
    logger.info(f"ContinuousFilterConv output shape: {output.shape}")
    assert output.shape == (10, num_features), "ContinuousFilterConv output shape mismatch"

    # Test backward pass
    output.sum().backward()
    logger.debug(f"Gradient shapes: x.grad={x.grad.shape}")

@profile
def test_message_block():
    logger.info("Testing MessageBlock")
    num_features = 64
    num_rbf = 20
    message_block = MessageBlock(num_features, num_rbf)

    s = Tensor(np.random.randn(10, num_features))
    v = Tensor(np.random.randn(10, num_features, 3))
    edge_index = Tensor(np.array([[0, 1, 2], [1, 2, 0]]))
    edge_attr = Tensor(np.random.rand(3, 3))

    logger.debug(f"Input shapes: s={s.shape}, v={v.shape}, edge_index={edge_index.shape}, edge_attr={edge_attr.shape}")
    delta_s, delta_v = message_block(s, v, edge_index, edge_attr)
    logger.info(f"MessageBlock output shapes: delta_s={delta_s.shape}, delta_v={delta_v.shape}")
    assert delta_s.shape == (10, num_features), "MessageBlock scalar output shape mismatch"
    assert delta_v.shape == (10, num_features, 3), "MessageBlock vector output shape mismatch"

    # Test backward pass
    (delta_s.sum() + delta_v.sum()).backward()
    logger.debug(f"Gradient shapes: s.grad={s.grad.shape}, v.grad={v.grad.shape}")

@profile
def test_update_block():
    logger.info("Testing UpdateBlock")
    num_features = 64
    update_block = UpdateBlock(num_features)

    s = Tensor(np.random.randn(10, num_features))
    v = Tensor(np.random.randn(10, num_features, 3))
    delta_s = Tensor(np.random.randn(10, num_features))
    delta_v = Tensor(np.random.randn(10, num_features, 3))

    logger.debug(f"Input shapes: s={s.shape}, v={v.shape}, delta_s={delta_s.shape}, delta_v={delta_v.shape}")
    s_out, v_out = update_block(s, v, delta_s, delta_v)
    logger.info(f"UpdateBlock output shapes: s_out={s_out.shape}, v_out={v_out.shape}")
    assert s_out.shape == (10, num_features), "UpdateBlock scalar output shape mismatch"
    assert v_out.shape == (10, num_features, 3), "UpdateBlock vector output shape mismatch"

    # Test backward pass
    (s_out.sum() + v_out.sum()).backward()
    logger.debug(f"Gradient shapes: s.grad={s.grad.shape}, v.grad={v.grad.shape}")

@profile
def test_linear():
    logger.info("Testing Linear")
    linear = Linear(64, 128)
    x = Tensor(np.random.randn(10, 64))

    logger.debug(f"Input shape: x={x.shape}")
    output = linear(x)
    logger.info(f"Linear output shape: {output.shape}")
    assert output.shape == (10, 128), "Linear output shape mismatch"

    # Test backward pass
    output.sum().backward()
    logger.debug(f"Gradient shapes: x.grad={x.grad.shape}, linear.W_linear.grad={linear.W_linear.grad.shape}")

@profile
def test_sequential():
    logger.info("Testing Sequential")
    seq = Sequential(
        Linear(64, 128),
        shifted_softplus,
        Linear(128, 64)
    )
    x = Tensor(np.random.randn(10, 64))

    logger.debug(f"Input shape: x={x.shape}")
    output = seq(x)
    logger.info(f"Sequential output shape: {output.shape}")
    assert output.shape == (10, 64), "Sequential output shape mismatch"

    # Test backward pass
    output.sum().backward()
    logger.debug(f"Gradient shape: x.grad={x.grad.shape}")

def test_edge_cases():
    logger.info("Testing edge cases")
    
    # Test with single-element input
    linear = Linear(1, 1)
    x = Tensor([[1.0]])
    output = linear(x)
    logger.debug(f"Single-element input output: {output.data}")

    # Test with empty input
    try:
        x = Tensor(np.array([]))
        output = linear(x)
    except Exception as e:
        logger.debug(f"Empty input raised exception as expected: {str(e)}")

if __name__ == "__main__":
    test_continuous_filter_conv()
    test_message_block()
    test_update_block()
    test_linear()
    test_sequential()
    test_edge_cases()
    logger.info("All tests completed!")
