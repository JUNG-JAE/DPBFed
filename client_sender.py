from socket import *
import sys
from os.path import exists

#
# from round_checker import current_round_checker
# import parameter as p

def send_file(host, port, filename):
    clientSock = socket(AF_INET, SOCK_STREAM)
    clientSock.connect((host, port))
    print('연결에 성공했습니다.')

    msg = filename.encode()
    length = len(msg)
    clientSock.sendall(length.to_bytes(4, byteorder="little"))
    clientSock.sendall(msg)

    data_transferred = 0

    if not exists(filename):
        print("no file")
        sys.exit()

    print("파일 %s 전송 시작" % filename)
    with open(filename, 'rb') as f:
        try:
            data = f.read(1024) # 1024바이트 읽는다
            while data: # 데이터가 없을 때까지
                data_transferred += clientSock.send(data) # 1024바이트 보내고 크기 저장
                data = f.read(1024) # 1024바이트 읽음
        except Exception as ex:
            print(ex)

    print("전송완료 %s, 전송량 %d" %(filename, data_transferred))
    clientSock.close()

# current_round = current_round_checker()
# send_file(p.SERVER_HOST, p.SERVER_PORT, "model/"+ str(1) +"/shard1.pt")