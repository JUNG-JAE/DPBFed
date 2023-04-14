# ------------ system library ------------ #
from socket import socket, AF_INET, SOCK_STREAM
import sys
from os.path import exists

# ------------ Custom library ------------ #
from conf.global_settings import SERVER_IP, SERVER_PORT


class Sender:
    def __init__(self, IP=SERVER_IP, PORT=SERVER_PORT):
        self.clientSock = socket(AF_INET, SOCK_STREAM)
        self.clientSock.connect((IP, PORT))
        print(f"Connect server ({IP}: {PORT})")

    def send_file(self, base_path, file_name):
        if not exists(f"{base_path}{file_name}"):
            print("no file")
            sys.exit()

        msg = file_name.encode()
        length = len(msg)
        self.clientSock.sendall(length.to_bytes(4, byteorder="little"))
        self.clientSock.sendall(msg)

        data_transferred = 0
        with open(f"{base_path}{file_name}", 'rb') as f:
            data = f.read(1024)
            while data:
                data_transferred += self.clientSock.send(data)
                data = f.read(1024)

        print(f"Sent {file_name} ({data_transferred}/bytes) to server")

    def close(self):
        self.clientSock.close()
        print("Socket closed")
