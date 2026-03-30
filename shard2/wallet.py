import binascii
# import sys # Crypto Error 발생시만 사용
# import crypto # Crypto Error 발생시만 사용
# sys.modules['Crypto'] = crypto # Crypto Error 발생시만 사용
import Crypto.Random
from Crypto.Signature import PKCS1_v1_5
from Crypto.PublicKey import RSA
from Crypto.Hash import SHA256


WALLET_LIST = []

def generate_wallet(inital = False):
    # if not inital:
    #     print("================= wallet created =================")
    # else:
    #     print("============= Genesis wallet created =============")

    random_gen = Crypto.Random.new().read
    private_key = RSA.generate(2048, random_gen)
    public_key = private_key.publickey()
    response = {
        'private_key': binascii.hexlify(private_key.export_key(format('PEM'))).decode('ascii'),
        'public_key': binascii.hexlify(public_key.export_key(format('PEM'))).decode('ascii'),
    }
    WALLET_LIST.append(response)
    return response

def sign(sender_private_key, transaction_dict):
    private_key_obj = RSA.importKey(binascii.unhexlify(sender_private_key))
    signer_obj = PKCS1_v1_5.new(private_key_obj)
    hash_obj = SHA256.new(str(transaction_dict).encode('utf-8'))
    return binascii.hexlify(signer_obj.sign(hash_obj)).decode('ascii')