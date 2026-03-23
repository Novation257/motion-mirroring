import socket

HOST = "0.0.0.0"
PORT = 5000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen()

print("Waiting for connection...")

conn, addr = server.accept()
print("Connected to:", addr)

while True:
    data = conn.recv(1024)

    if not data:
        break

    message = data.decode().strip()

    yaw, pitch, roll, flex = message.split(",")

    print("Yaw:", yaw)
    print("Pitch:", pitch)
    print("Roll:", roll)
    print("Flex:", flex)
    print("----------------")