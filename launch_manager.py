import subprocess
import os

processes = {}

def launch_process(key, command):
    stop_process(key)

    ros_setup = "/opt/ros/humble/setup.bash"
    workspace_setup = "/home/aryan/rn_prototype/install/setup.bash"

    full_cmd = f"bash -c 'source {ros_setup} && source {workspace_setup} && {command}'"
    print(f"[DEBUG] Launching command: {full_cmd}")

    try:
        with open("launch_log.txt", "a") as log:
            log.write(f"\n[COMMAND]: {full_cmd}\n")
            proc = subprocess.Popen(
                full_cmd,
                shell=True,
                stdout=log,
                stderr=log,
                executable="/bin/bash"
            )
            processes[key] = proc
            return proc
    except Exception as e:
        print(f"[ERROR] Failed to launch {key}: {e}")
        return None

def stop_process(key):
    if key in processes:
        try:
            processes[key].terminate()
            processes[key].wait(timeout=5)
        except Exception as e:
            print(f"[ERROR] Failed to stop {key}: {e}")
        finally:
            del processes[key]

def start_amr():
    return launch_process("amr", "ros2 launch rn_hardware amr_bringup.launch.py")

def stop_amr():
    stop_process("amr")

def start_mapping():
    return launch_process("mapping", "ros2 launch rn_hardware online_async_launch.py")

def start_navigation():
    return launch_process("navigation", "ros2 launch rn_hardware goal_nav_launch.py")
