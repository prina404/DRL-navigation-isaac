import jax

from jax_loader import get_policy
import numpy as np
import torch

from torch_loader import build_torch_policy_from_jax_params

def build_jax_obs():
    rand_pos = np.random.rand(1, 13)
    rand_vel = np.random.rand(1, 13)
    rand_act = np.random.rand(1, 13)

    state = np.vstack((rand_pos, rand_vel, rand_act)).T.flatten()
    ang_vel = np.random.rand(3)
    cmd = np.random.rand(3)
    gravity_vec = np.random.rand(3)
    return np.concatenate((state, ang_vel, cmd, gravity_vec))

def call_policy(policy, params, obs):
    return jax.device_get(policy.apply(params, obs))


def call_torch_policy(model, obs):
    obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32))
    with torch.no_grad():
        out = model(obs_t)
    return out.detach().cpu().numpy()


if __name__ == "__main__":
    policy, params = get_policy()
    torch_policy = build_torch_policy_from_jax_params(
        params,
        action_space=12,
        policy_mean_abs_clip=policy.policy_mean_abs_clip,
        device="cpu",
    )

    r1 = build_jax_obs()
    r2 = build_jax_obs()

    res_1 = call_policy(policy, params, r1)
    res_2 = call_policy(policy, params, r2)
    torch_res_1 = call_torch_policy(torch_policy, r1)
    torch_res_2 = call_torch_policy(torch_policy, r2)

    atol = 1e-5
    rtol = 1e-4

    max_abs_1 = np.max(np.abs(res_1 - torch_res_1))
    max_abs_2 = np.max(np.abs(res_2 - torch_res_2))

    same_1 = np.allclose(res_1, torch_res_1, atol=atol, rtol=rtol)
    same_2 = np.allclose(res_2, torch_res_2, atol=atol, rtol=rtol)

    print("JAX output 1:", res_1)
    print("Torch output 1:", torch_res_1)
    print("max_abs_diff_1:", max_abs_1)
    print("allclose_1:", same_1)

    print("JAX output 2:", res_2)
    print("Torch output 2:", torch_res_2)
    print("max_abs_diff_2:", max_abs_2)
    print("allclose_2:", same_2)

    if not (same_1 and same_2):
        raise AssertionError(
            f"JAX/Torch mismatch: sample1 max_abs={max_abs_1}, sample2 max_abs={max_abs_2}"
        )

    print("JAX and Torch outputs match within tolerance.")