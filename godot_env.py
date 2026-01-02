import socket
import json
import numpy as np
import time

class GodotEnv:
    def __init__(self, host='127.0.0.1', port=4242):
        self.host = host
        self.port = port
        self.sock = None
        self.buffer_size = 8192
        
        # --- DIMENSIONS ---
        # 34 (Paper Params) + 4 (Contact Sensors) = 38 Total
        # [0]      : Raycast Ground Distance
        # [1:4]    : Gravity / Body Rotation (3)
        # [4:16]   : Joint Positions (12)
        # [16:19]  : Angular Velocity (3)
        # [19:22]  : Linear Acceleration (3)
        # [22:34]  : Joint Velocities (12)
        # [34:37]  : Foot Contacts (4)

        self.state_dim = 38 
        self.action_dim = 12 
        self.max_motor_speed = 2.0 

        self.last_lin_vel = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self.target_height = 0.5  # Reference height for the robot body

    def connect(self):
        try:
            if self.sock: self.sock.close()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5.0) 
            self.sock.connect((self.host, self.port))
            print(f"Connected to Godot at {self.host}:{self.port}")
            time.sleep(0.5) 
            return True
        except:
            return False

    def reset(self):
        state = None

        # 🔹 Reset episode-dependent buffers
        self.last_lin_vel = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)

        while state is None:
            if not self.sock:
                while not self.connect():
                    time.sleep(1)
            try:
                # First step with zero action
                state, _, _ = self.step(np.zeros(self.action_dim, dtype=np.float32))
                if state is None:
                    self.close()
                    time.sleep(1)
            except:
                self.close()
                time.sleep(1)

        return state


    def step(self, action):
        if not self.sock: return None, 0.0, True
        try:
            # Convert action to list for JSON and ensure clipping
            scaled_action = (np.clip(action, -1.0, 1.0) * self.max_motor_speed).tolist()
            msg = json.dumps({"velocities": scaled_action}) + "\n"
            self.sock.sendall(msg.encode('utf-8'))

            buffer = ""
            while "\n" not in buffer:
                chunk = self.sock.recv(self.buffer_size)
                if not chunk: raise ConnectionError("Socket closed")
                buffer += chunk.decode('utf-8')
            
            data = json.loads(buffer.strip().split("\n")[-1])
            state_array = self._process_state_research(data["state"])
            
            # --- REWARD SHAPING (Research-Grade Improvements) ---
            # Explicitly use float32 for all reward components to avoid PyTorch Double errors
            raw_reward = np.float32(data["reward"])
            
            # 1. Penalize high vertical velocity
            lin_vel = np.array(data["state"].get("linear_velocity", [0,0,0]), dtype=np.float32)
            v_y_penalty = np.float32(-0.1 * (lin_vel[1] ** 2))
            
            # 2. Penalize height error (using Raycast)
            ground_dist = np.float32(state_array[0])
            height_error = abs(ground_dist - self.target_height)
            height_penalty = -1.0 * min(height_error, 0.3)

            # 3. CENTER OF MASS (CoM) STABILITY
            body_pos = np.array(data["state"]["body_position"], dtype=np.float32)
            foot_positions = np.array(data["state"].get("foot_positions", []), dtype=np.float32)
            contacts = np.array(data["state"].get("foot_contacts", [False]*4))
            
            com_reward = np.float32(0.0)
            if len(foot_positions) == 4 and np.any(contacts):
                active_feet = foot_positions[contacts]
                support_center = np.mean(active_feet, axis=0)
                dist_to_com = np.linalg.norm(body_pos[[0, 2]] - support_center[[0, 2]])
                com_reward = np.float32(-1.0 * dist_to_com)

            # 4. FOUR-LEGGED ENCOURAGEMENT
            num_contacts = np.sum(contacts)
            forward_vel = lin_vel[2]  # or correct axis
            contact_reward = np.float32(0.05 * num_contacts * max(forward_vel, 0.0))

            # 5. ACTION SMOOTHING
            jitter_penalty = np.float32(-0.01 * np.sum(np.square(action - self.last_action)))
            self.last_action = np.copy(action).astype(np.float32)

            shaped_reward = (raw_reward + 
                             v_y_penalty + 
                             height_penalty + 
                             com_reward + 
                             contact_reward + 
                             jitter_penalty)
            
            # Final return cast to ensure Python float doesn't promote it to double
            if abs(shaped_reward) > 100:
                print("WARNING: Large reward", shaped_reward)
            return state_array, float(shaped_reward), bool(data["done"])
            
        except Exception as e:
            print(f"Step Error: {e}")
            self.close()
            return None, 0.0, True

    def _process_state_research(self, raw):
        # 1. Ground Distance via Raycast (Index 0)
        ground_dist = [np.float32(raw.get("ground_distance", raw["body_position"][1]))]

        # 2. Rotation/Gravity (Index 1:4)
        gravity = [np.float32(x) for x in raw["body_rotation"]]

        # 3. Joint Positions + Feature Engineering (Index 4:16)
        q = np.array(raw["joint_angles"], dtype=np.float32)
        q_feat = list(q)
        q_feat[1] = q[1] - q[4] 
        q_feat[2] = q[2] - q[5] 
        q_feat[7] = q[7] - q[10] 
        q_feat[8] = q[8] - q[11] 

        # 4. Angular Velocity (Index 16:19)
        ang_vel = [np.float32(x) for x in raw.get("angular_velocity", [0,0,0])]

        # 5. Linear Acceleration (Index 19:22)
        curr_vel = np.array(raw.get("linear_velocity", [0,0,0]), dtype=np.float32)
        dt = 0.016  # ideally receive from Godot
        lin_accel = np.clip(
            (curr_vel - self.last_lin_vel) / dt,
            -10.0, 10.0
        ).tolist()
        self.last_lin_vel = curr_vel

        # 6. Joint Velocities (Index 22:34)
        q_vel = [np.float32(x) for x in raw.get("joint_velocities", [0]*12)]

        # 7. Foot Contacts (Index 34:37)
        # REQUIRED ORDER from Godot:
        # [FrontRight, FrontLeft, BackRight, BackLeft]
        contacts_raw = raw.get("foot_contacts", [False]*4)
        contacts = [
            1.0 if contacts_raw[0] else 0.0,  # FR
            1.0 if contacts_raw[1] else 0.0,  # FL
            1.0 if contacts_raw[2] else 0.0,  # BR
            1.0 if contacts_raw[3] else 0.0   # BL
        ]

        # Combined array as float32
        return np.array(ground_dist + gravity + q_feat + ang_vel + lin_accel + q_vel + contacts, dtype=np.float32)

    def close(self):
        if self.sock:
            try: self.sock.close()
            except: pass
            self.sock = None