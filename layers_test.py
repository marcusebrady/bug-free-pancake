import numpy as np
from autograd import Tensor
from layers import ContinuousFilterConv, MessageBlock, UpdateBlock, rbf_expansion, Linear, Sequential, shifted_softplus
from functools import partial
import cProfile
import pstats
import io
import logging


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

    num_edges = 3
    x = Tensor(np.random.randn(10, num_features))
    edge_index = Tensor(np.array([[0, 1, 2], [1, 2, 0]]))
    edge_attr = Tensor(np.random.rand(num_edges, 3))

    logger.info(f"x shape: {x.shape}")
    logger.info(f"edge_index shape: {edge_index.shape}")
    logger.info(f"edge_attr shape: {edge_attr.shape}")

    # Add more logging
    rbf_features = rbf_expansion(edge_attr, num_rbf, cutoff)
    logger.info(f"rbf_features shape: {rbf_features.shape}")
    
    filters = cfconv.filter_network(rbf_features)
    logger.info(f"filters shape after network: {filters.shape}")
    
    output = cfconv(x, edge_index, edge_attr)
    logger.info(f"ContinuousFilterConv output shape: {output.shape}")
    assert output.shape == (num_edges, 3, num_features), f"ContinuousFilterConv output shape mismatch. Expected {(num_edges, 3, num_features)}, got {output.shape}"

@profile
def test_message_block():
    logging.info("Testing MessageBlock")
    num_features = 64
    num_rbf = 20
    cutoff = 5.0  
    message_block = MessageBlock(num_features, num_rbf, cutoff) 

    num_nodes = 10
    num_edges = 3
    s = Tensor(np.random.randn(num_nodes, num_features))
    v = Tensor(np.random.randn(num_nodes, num_features, 3))
    edge_index = Tensor(np.array([[0, 1, 2], [1, 2, 0]]))
    edge_attr = Tensor(np.random.rand(num_edges, 3))

    logging.info(f"s shape: {s.shape}")
    logging.info(f"v shape: {v.shape}")
    logging.info(f"edge_index shape: {edge_index.shape}")
    logging.info(f"edge_attr shape: {edge_attr.shape}")

    
    s_j = s[Tensor(edge_index.data[0])]
    v_j = v[Tensor(edge_index.data[0])]
    phi_output = message_block.phi(s_j)
    filters = message_block.conv_filter(s_j, edge_index, edge_attr)
    message_block._debug_shapes(s, v, edge_index, edge_attr)

    delta_s, delta_v = message_block(s, v, edge_index, edge_attr)
    
    logging.info(f"MessageBlock output shapes: delta_s={delta_s.shape}, delta_v={delta_v.shape}")
    assert delta_s.shape == (num_nodes, num_features), f"MessageBlock scalar output shape mismatch. Expected {(num_nodes, num_features)}, got {delta_s.shape}"
    assert delta_v.shape == (num_nodes, num_features, 3), f"MessageBlock vector output shape mismatch. Expected {(num_nodes, num_features, 3)}, got {delta_v.shape}"

@profile
def test_update_block():
    logger.info("Testing UpdateBlock")
    num_features = 64
    update_block = UpdateBlock(num_features)

    s = Tensor(np.random.randn(10, num_features))
    v = Tensor(np.random.randn(10, num_features, 3))
    delta_s = Tensor(np.random.randn(10, num_features))
    delta_v = Tensor(np.random.randn(10, num_features, 3))

    s_out, v_out = update_block(s, v, delta_s, delta_v)
    logger.info(f"UpdateBlock output shapes: s_out={s_out.shape}, v_out={v_out.shape}")
    assert s_out.shape == (10, num_features), "UpdateBlock scalar output shape mismatch"
    assert v_out.shape == (10, num_features, 3), "UpdateBlock vector output shape mismatch"

if __name__ == "__main__":
    test_continuous_filter_conv()
    test_message_block()
    test_update_block()
    logger.info("All tests passed!")
