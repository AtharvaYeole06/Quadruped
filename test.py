import sys
import time
import mujoco
import mujoco.viewer

# Ensure python can find your custom package
sys.path.insert(0, "/home/atharvayeole/Projects/vision4leg")

from vision4leg.envs.a1_mujoco_env import A1MujocoEnv
from stable_baselines3 import PPO


def main():
    # 1. Initialize environment (leave render_mode out since it does nothing)
    env = A1MujocoEnv()

    # 2. Load your trained model weights
    model_path = "a1_mujoco_ppo"
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    # 3. Reset the environment to initialize everything
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    # --- THE FIX: Extract the hidden MuJoCo pointers ---
    mj_model = env._robot._model
    mj_data = env._robot._data

    print("Launching native MuJoCo passive viewer window...")

    # 4. Open the native MuJoCo interactive viewer loop
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        print("Window opened successfully! Close the window or press Ctrl+C to stop.")

        while viewer.is_running():
            step_start = time.time()

            # Predict action from your 1M step policy
            action, _states = model.predict(obs, deterministic=True)

            # Step the environment physics forward
            step_results = env.step(action)
            if len(step_results) == 5:
                obs, reward, terminated, truncated, info = step_results
                done = terminated or truncated
            else:
                obs, reward, done, info = step_results

            # Force the viewer window to sync with the new physics state
            viewer.sync()

            if done:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]

            # Standard delay to keep the playback speed human-readable
            time_until_next_step = 0.01 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
