import socket, threading
import parameter as p
import concurrent.futures

shard_model_list = []

lock = threading.Lock()


def binder(client_socket, addr):
    print('클라이언트 [ {0} ] 에서 접속'.format(addr))
    try:
        # while True:
        msg = client_socket.recv(4)
        length = int.from_bytes(msg, "little")
        msg = client_socket.recv(length)

        filename = msg.decode()
        # if filename == '/exit':
        #     print("-> {0} 클라이언트에 의해 중단".format(client_socket.client_address[0]))

        data = client_socket.recv(1024)

        data_transferred = 0

        with open(filename, 'wb') as f:
            try:
                while data:
                    f.write(data)
                    data_transferred += len(data)
                    data = client_socket.recv(1024)
            except Exception as ex:
                print(ex)

        print('송신완료{0}, 송신량{1}'.format(filename, data_transferred))

        if filename not in shard_model_list:
            lock.acquire()
            shard_model_list.append(filename)
            lock.release()
            print(shard_model_list)

        if len(shard_model_list) == p.SHARD_NUM * p.UPLOAD_MODEL_NUM:
            print("All shard update model")
            return "exit"
        else:
            return "continue"

    except Exception as e:
        print("error : ", e)
    finally:
        client_socket.close()


def runServer():
    print("-> model server start")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((p.HOST, p.PORT))
    server_socket.listen()

    try:
        while True:
            client_socket, addr = server_socket.accept()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(binder, client_socket, addr)
                return_value = future.result()
                if return_value == "exit":
                    break
    except:
        print("file server closed")
    finally:
        server_socket.close()


"""
def runServer():
    print("-> file server start")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # server_socket.bind((p.HOST, p.PORT))
    server_socket.bind(('127.0.0.1', 9060))
    server_socket.listen()

    try:
        while True:
            client_socket, addr = server_socket.accept()
            th = threading.Thread(target=binder, args=(client_socket, addr, server_socket))
            th.start()
    except:
        print("file server closed")
    finally:
        server_socket.close()
"""
