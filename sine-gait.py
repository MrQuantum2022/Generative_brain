import time
import socket
import json
import numpy as np
from velocity_sine_gait import VelocitySineGait  # your class

HOST = "127.0.0.1"
PORT = 4242

def main():
    gait = VelocitySineGait(dt=0.016)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    print("Connected to Godot for gait test")

    try:
        while True:
            action = gait.step().tolist()
            msg = json.dumps({"velocities": action}) + "\n"
            sock.sendall(msg.encode("utf-8"))

            # Receive state (we ignore it, but must read it)
            buffer = ""
            while "\n" not in buffer:
                buffer += sock.recv(8192).decode("utf-8")

            time.sleep(0.016)  # ~60 Hz

    except KeyboardInterrupt:
        print("Stopping gait test")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
