from socket import *
import sys
from os.path import exists


class sendServer:
    def __init__(self, HOST, PORT):
        self.clientSock = socket(AF_INET, SOCK_STREAM)
        self.clientSock.connect((HOST, PORT))
        print('[ {0}: {1} ]연결에 성공했습니다.'.format(HOST, PORT))

    def send_file(self, filename):
        msg = filename.encode()
        length = len(msg)
        self.clientSock.sendall(length.to_bytes(4, byteorder="little"))
        self.clientSock.sendall(msg)

        data_transferred = 0

        if not exists(filename):
            print("no file")
            sys.exit()

        print("파일 {0} 전송 시작".format(filename))
        with open(filename, 'rb') as f:
            try:
                data = f.read(1024)
                while data:
                    data_transferred += self.clientSock.send(data)
                    data = f.read(1024)
            except Exception as ex:
                print(ex)

        print("전송완료 {0}, 전송량 {1}".format(filename, data_transferred))

"""
sendServer(HOST, PORT)

for address in host_list:
    for filename in file_list:
        shard = sendServer(address["ip"], address["port"])
        if filename == ".DS_Store":
            pass
        shard.send_file("model/" + filename)
        shard.clientSock.close()
        print("socket closed")
"""