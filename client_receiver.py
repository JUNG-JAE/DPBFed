import socket, threading
import parameter as p
import concurrent.futures

shard_model_list = []

trigger = True

def binder(client_socket, addr, migrate_mode):
    global trigger
    # 커넥션이 되면 접속 주소가 나온다.
    print('Connected by', addr)
    try:
        # while True:
        msg = client_socket.recv(4)
        length = int.from_bytes(msg, "little")
        msg = client_socket.recv(length)
        filename = msg.decode()

        data = client_socket.recv(1024)
        data_transferred = 0

        if not data:
            print('파일 %s 가 서버에 존재하지 않음' % filename)
            # sys.exit()

        with open(filename, 'wb') as f:  # 현재dir에 filename으로 파일을 받는다
            try:
                while data:  # 데이터가 있을 때까지
                    f.write(data)  # 1024바이트 쓴다
                    data_transferred += len(data)
                    data = client_socket.recv(1024)  # 1024바이트를 받아 온다
            except Exception as ex:
                print(ex)
        print('송신완료[%s], 송신량[%d]' % (filename, data_transferred))

        # if filename[8:] not in shard_model_list:
        #     shard_model_list.append(filename[8:])
        #     print(shard_model_list)

        if filename not in shard_model_list:
            shard_model_list.append(filename)
            print(shard_model_list)

        if migrate_mode & len(shard_model_list) == 1:
            shard_model_list.clear()
            print("receive migration info")
            # server_socket.close()
            # break
            return "exit"


        # aggregation model을 추가적으로 받기위해 +1을 함
        if len(shard_model_list) == p.SHARD_NUM + 1:
            shard_model_list.clear()
            print("All data receive form server")
            # server_socket.close()
            # break
            return "exit"

        # if "model/" + str(1) + "/aggregation.pt" in shard_model_list:
        #     shard_model_list.clear()
        #     print("All data receive form server")
        #     return "exit"

    except:
        print("except : ", addr)
    finally:
        client_socket.close()


def runReceiver(migrate_mode=False):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((p.SHARD_HOST, p.SHARD_PORT))
    server_socket.listen()
    print("client receiver start")
    try:
        while True:
            client_socket, addr = server_socket.accept()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(binder, client_socket, addr, migrate_mode)
                return_value = future.result()
                if return_value == "exit":
                    break
    except:
        print("server closed")
    finally:
        server_socket.close()

# runReceiver()