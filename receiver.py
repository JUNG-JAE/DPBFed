import socket
import threading
import concurrent.futures
from conf.global_settings import LOG_DIR, DATA_TYPE, SHARD_IP, SHARD_PORT, NUM_OF_SHARD


shard_model_list = []
lock = threading.Lock()


def binder(client_socket, addr, base_path):
    try:
        msg = client_socket.recv(4)
        length = int.from_bytes(msg, "little")
        msg = client_socket.recv(length)
        filename = msg.decode()

        data = client_socket.recv(1024)
        data_transferred = 0

        print(f"Server receive file {filename} from {addr}")

        with open(base_path + filename, 'wb') as f:
            try:
                while data:
                    f.write(data)
                    data_transferred += len(data)
                    data = client_socket.recv(1024)
            except Exception as ex:
                print(ex)

        with lock:
            if filename not in shard_model_list:
                shard_model_list.append(filename)

        if len(shard_model_list) == NUM_OF_SHARD + 1:
            print(f"All file received {shard_model_list}")
            return "exit"
        else:
            return "continue"

    except Exception as e:
        print("error : ", e)
    finally:
        client_socket.close()


def run_receiver(base_path):
    print("-> Receiver listening")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((SHARD_IP, SHARD_PORT))
    server_socket.listen()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        while True:
            client_socket, addr = server_socket.accept()

            future = executor.submit(binder, client_socket, addr, base_path)
            return_value = future.result()

            if return_value == "exit":
                break

    server_socket.close()


