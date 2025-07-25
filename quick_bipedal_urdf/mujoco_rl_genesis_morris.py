
import time
import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
import argparse

NUM_MOTOR = 6

def get_sensor_data(model, data, sensor_name):
    """Dynamically retrieve sensor data by name."""
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id == -1:
        raise ValueError(f"Sensor '{sensor_name}' not found in model!")
    start_idx = model.sensor_adr[sensor_id]
    dim = model.sensor_dim[sensor_id]
    return data.sensordata[start_idx : start_idx + dim]

def calculate_com_in_base_frame(model, data, base_body_id):
    total_mass = 0.0
    com_sum = np.zeros(3)
    base_pos = data.xipos[base_body_id]
    base_rot = data.ximat[base_body_id].reshape(3, 3)
    for i in range(model.nbody):
        mass = model.body_mass[i]
        world_com = data.xipos[i]
        local_com = world_com - base_pos
        local_com = base_rot.T @ local_com
        com_sum += mass * local_com
        total_mass += mass
    return com_sum / total_mass

def quat_rotate_inverse(q, v):
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    term1 = 2.0 * np.square(q_w) - 1.0
    term1_expanded = np.expand_dims(term1, axis=-1)
    a = v * term1_expanded
    b = np.cross(q_vec, v) * np.expand_dims(q_w, axis=-1) * 2.0
    dot_product = np.sum(q_vec * v, axis=-1)
    dot_product_expanded = np.expand_dims(dot_product, axis=-1)
    c = q_vec * dot_product_expanded * 2.0
    return a - b + c

def get_gravity_orientation(model, data, quaternion_sensor_name="orientation"):
    quaternion = get_sensor_data(model, data, quaternion_sensor_name)
    gravity_world = np.array([0, 0, -1])
    if quaternion.shape == (4,):
        quaternion = quaternion.reshape(1, 4)
        gravity_world = gravity_world.reshape(1, 3)
        result = quat_rotate_inverse(quaternion, gravity_world)[0]
    else:
        gravity_world = np.broadcast_to(gravity_world, quaternion.shape[:-1] + (3,))
        result = quat_rotate_inverse(quaternion, gravity_world)
    return result

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        policy = torch.jit.load(policy_path)
        xml_path = config["xml_path"]
        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        pos_action_scale = config["pos_action_scale"]
        vel_action_scale = config["vel_action_scale"]
        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        one_step_obs_size = config["one_step_obs_size"]
        obs_buffer_size = config["obs_buffer_size"]
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    target_dof_pos = default_angles.copy()
    target_dof_vel = np.zeros(6)
    action = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    base_body_id = 1

    # Define DOF names (adjust based on your XML)
    dof_names = ["R_thigh","R_calf","L_thigh","L_calf","R_wheel","L_wheel"]

    counter = 0
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # mujoco.mj_step(m, d)
            qpos = np.array([get_sensor_data(m, d, f"{name}_pos")[0] for name in dof_names])
            qvel = np.array([get_sensor_data(m, d, f"{name}_vel")[0] for name in dof_names])
            # counter += 1
            # if counter % control_decimation == 0 and counter > 0:
            # Create observation
            dof_pos = qpos[:4]
            default_pos = default_angles[:4]
            ang_vel_b = get_sensor_data(m, d, "base_ang_vel")  # Direct base-frame angular velocity
            gravity_b = get_gravity_orientation(m, d, "imu_quat")
            cmd_vel = np.array(config["cmd_init"], dtype=np.float32)
            obs_list = [
                ang_vel_b * ang_vel_scale,  # 3
                gravity_b,  # 3
                cmd_vel * cmd_scale,  # 4
                (dof_pos - default_pos) * dof_pos_scale,  # 4
                qvel * dof_vel_scale,  # 6
                action.astype(np.float32)  # 6
            ]
            print(ang_vel_b)
            obs_list = [torch.tensor(obs, dtype=torch.float32) if isinstance(obs, np.ndarray) else obs for obs in obs_list]
            obs_tensor_buf = torch.zeros((1, one_step_obs_size * obs_buffer_size))
            obs = torch.cat(obs_list, dim=0).unsqueeze(0)
            obs_tensor = torch.clamp(obs, -100, 100)

            obs_tensor_buf = torch.cat([
                obs_tensor,
                obs_tensor_buf[:, :obs_buffer_size * one_step_obs_size - one_step_obs_size]
            ], dim=1)

            # Policy inference
            action = policy(obs_tensor_buf).detach().numpy().squeeze()

            # # Transform action to target_dof_pos
            # target_dof_pos = np.array([action[0], action[1], action[2], action[3], 0, 0]) * pos_action_scale + default_angles
            # target_dof_vel = np.array([0, 0, 0, 0, action[4], action[5]]) * vel_action_scale
            # # Get joint positions and velocities using sensor names
            # qpos = np.array([get_sensor_data(m, d, f"{name}_pos")[0] for name in dof_names])
            # qvel = np.array([get_sensor_data(m, d, f"{name}_vel")[0] for name in dof_names])

            # tau = pd_control(target_dof_pos, qpos, kps, target_dof_vel, qvel, kds)
            # d.ctrl[:] = tau

                       # update action
            target_dof_pos = action[0:4] * pos_action_scale + default_pos
            target_dof_vel = action[4:6] * vel_action_scale
            # print("act:", act)
            for i in range(4):
                d.ctrl[i] = target_dof_pos[i]

            d.ctrl[4] = target_dof_vel[0]
            d.ctrl[5] = target_dof_vel[1]

            for i in range(5):
                mujoco.mj_step(m, d)

            viewer.sync()
            time_until_next_step = 0.01 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
