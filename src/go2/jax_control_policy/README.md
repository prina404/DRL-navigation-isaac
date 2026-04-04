## Disclaimer

The code and model in this folder is stored just for archival  and reference purposes. It's derived from Dyuman and Nico's respective works: 

```bibtex
@INPROCEEDINGS{adityaRobustQuad2025,
  author={Aditya, Dyuman and Huang, Junning and Bohlinger, Nico and Kicki, Piotr and Walas, Krzysztof and Peters, Jan and Luperto, Matteo and Tateo, Davide},
  booktitle={2025 European Conference on Mobile Robots (ECMR)}, 
  title={Robust Localization, Mapping, and Navigation for Quadruped Robots}, 
  year={2025},
  doi={10.1109/ECMR65884.2025.11163249}}
}

@article{bohlinger2024one,
  title={One policy to run them all: an end-to-end learning approach to multi-embodiment locomotion},
  author={Bohlinger, Nico and Czechmanowski, Grzegorz and Krupka, Maciej and Kicki, Piotr and Walas, Krzysztof and Peters, Jan and Tateo, Davide},
  journal={arXiv preprint arXiv:2409.06366},
  year={2024}
}
```
---
The original JAX model expects data w/ a specific format and normalization.

expected jax inputs:
```
JOINTS -----
  [[p1, p2, p3, ...], # (13,)
   [v1, v2, v3, ...], # (13,)
   [a1, a2, a3, ...]] # (13,) a = prev_action
transposed: 
  [[p1, v1, a1], 
   [p2, v2, a2],
   [p3, v3, a3], ...]
flattened: [[p1, v1, a1], [p2, v2, a2], [p3, v3, a3], ...] # (39,)

ANG VEL --------
[yaw, pitch, roll]

CMD VEL --------
[x, y, theta]

GRAVITY VEC -----
[x_g, y_g, z_g]
```

Original code to build the observation vector:
```
qpos = (self.joint_positions - self.nominal_joint_positions) / 3.14
qvel = self.joint_velocities / self.max_joint_velocities
previous_action = self.previous_action / 3.14
qpos_qvel_previous_action = np.vstack((qpos, qvel, previous_action)).T.flatten()

ang_vel = self.angular_velocity / 10.0
orientation_quat_inv = R.from_quat(self.orientation).inv()
projected_gravity_vector = orientation_quat_inv.apply(np.array([0.0, 0.0, -1.0]))

observation = np.concatenate([
    qpos_qvel_previous_action, ang_vel,
    [self.x_goal_velocity, self.y_goal_velocity, self.yaw_goal_velocity],
    projected_gravity_vector
])

action = jax.device_get(self.policy.apply(self.policy_state.params, observation)) 
```