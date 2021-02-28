import torch
import pyspiel

def preprocess_state_before_store(step):
    # Minigrid
    if isinstance(step.state, tuple):
        step.state = step.state.obs
        step.next_state = step.next_state.obs

    # Open Spiel
    elif isinstance(step.state, pyspiel.State):
        t = step.state.observation_tensor(0)
        next_t = step.next_state.observation_tensor(0)
        t = torch.tensor(t)
        next_t = torch.tensor(next_t)
        t = t.reshape(11, 8, 8)
        next_t = next_t.reshape(11, 8, 8)
        step.state = t.detach().numpy()
        step.next_state = next_t.detach().numpy()

    # MNIST
    elif isinstance(step.state, torch.Tensor):
        assert isinstance(step.next_state, torch.Tensor)
        if step.state.dim() == 4:
            assert step.state.shape == step.next_state.shape == (1, 1, 64, 64)
            step.state = step.state[0]
            step.next_state = step.next_state[0]
        step.state = step.state.detach().numpy()
        step.next_state = step.next_state.detach().numpy()
    return step

def preprocess_state_before_forward(state):
    if isinstance(state, tuple):
        return state.obs
    elif isinstance(state, pyspiel.State):
        t = state.observation_tensor(0)
        t = torch.tensor(t)
        t = t.reshape(11, 8, 8)
        return t
    else:
        return state