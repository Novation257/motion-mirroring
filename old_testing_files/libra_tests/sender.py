import socket
import time
SERVER_IP = "10.131.96.32"
PORT = 5000

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((SERVER_IP, PORT))

print("Connected to server")

while True:
	yaw = 10.0
	pitch = 2.0
	roll = 5.0

	message = f"{yaw},{pitch},{roll}\n"
	client.send(message.encode())
	print("Sent: ", message)
	time.sleep(0.1)

