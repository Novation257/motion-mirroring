import socket
HOST = "0.0.0.0"
PORT = 5000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen()

print("Waiting for connection...")
conn, addr = server.accept()

print("Connected to ", addr)

# default safe position
angle = 90

buffer = b' '

while True:
        data = conn.recv(1024)
        if not data:
            continue
        buffer += data
        while b'\n' in buffer:
            line, buffer = buffer.split(b'\n', 1)
            message = line.decode().strip()
            parts = message.split(',')

            if message == "STOP":
                print("STOP received: holding position")
                continue

            if len(parts) == 4:
                try:
                    yaw, pitch, roll, flex = map(float, parts)
                    pitch = float(pitch)
                    angle = max(0, min(180, (pitch + 45)*2))
                    print(f"Pitch: {pitch:.2f} : Servo: {angle:.1f}")

                except Exception as e:
                    print("Error:", e)
            else:
                print("Control:", message)
