"""Layer selectors of the Workflow Adapter (paper Section 3.2, Appendix D.1)."""

import pytest
import torch

from src.workflow_adapter.selectors import AttentionLayerAgentSelector, LayerAgentSelector

CPU = torch.device("cpu")


@pytest.mark.parametrize("is_first_layer", [True, False])
def test_attention_selector_outputs_distribution(is_first_layer):
    selector = AttentionLayerAgentSelector(d_model=32, num_heads=4, device=CPU, is_first_layer=is_first_layer)
    state, agents = torch.randn(10, 32), torch.randn(7, 32)
    prev = None if is_first_layer else agents[[0, 2]]
    log_probs, probs = selector(state, agents, prev)
    assert probs.shape == (1, 7)
    assert torch.allclose(probs.sum(dim=1), torch.ones(1))
    assert torch.allclose(log_probs.exp(), probs, atol=1e-6)


def test_non_first_layer_requires_upstream_context():
    selector = AttentionLayerAgentSelector(d_model=32, num_heads=4, device=CPU, is_first_layer=False)
    with pytest.raises(ValueError):
        selector(torch.randn(10, 32), torch.randn(7, 32))


def test_temperature_sharpens_distribution():
    torch.manual_seed(0)
    selector = AttentionLayerAgentSelector(d_model=32, num_heads=4, device=CPU, is_first_layer=True).eval()
    state, agents = torch.randn(10, 32), torch.randn(7, 32)
    _, p_hot = selector(state, agents)
    selector.temperature = 0.1
    _, p_cold = selector(state, agents)
    assert p_cold.max() > p_hot.max()


@pytest.mark.parametrize("cls,kwargs", [
    (AttentionLayerAgentSelector, dict(d_model=32, num_heads=4)),
    (LayerAgentSelector, dict(input_dim=32, hidden_dim=16)),
])
def test_load_restores_parameters_in_place(tmp_path, cls, kwargs):
    selector = cls(device=CPU, is_first_layer=True, **kwargs)
    path = tmp_path / "selector.pth"
    selector.save(str(path))
    reference = [p.detach().clone() for p in selector.parameters()]
    with torch.no_grad():
        for p in selector.parameters():
            p.add_(1.0)
    selector.load(str(path))
    assert all(torch.equal(a, b) for a, b in zip(reference, selector.parameters()))
