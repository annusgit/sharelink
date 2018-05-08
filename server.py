
"""
    Author: Annus Zulfiqar
    Date: 5th May, 2018
"""

from __future__ import print_function
from __future__ import division
from utilities import recursive_zip
import zipfile
import socket

server = ('127.0.0.1', 50000)
this_machine = ('127.0.0.1', 50001)


def main():

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(server)
    print('log: ready to accept a tcp connection...')
    sock.listen(1)
    client_sock, addr = sock.accept()    # connection established now!!!
    print('connection from {} successful!'.format(addr))

    # this is what we want to send
    # _object_ = '/home/annus/PycharmProjects/instruments_chair_project'
    _object_ = '/home/annus/PycharmProjects/sharelink/warofchange.mp4'

    # we'll zip it and send it as a file, then at the other end, we'll unzip it and save it
    # here we read and zip files and folders
    with zipfile.ZipFile('source.zip', 'w') as zipobject:
        recursive_zip(entity=_object_, zipfileobject=zipobject)

    # now let's zip it to send over network
    with open('source.zip', 'r') as source:
        # let's send this zip file
        buf_size = 1024; data = source.read(buf_size); count = 1
        print('log: sending now...')
        while data:
            client_sock.send(data)
            data = source.read(buf_size)
            count += 1
            pass
        print('log: sent {} kbs to the client'.format(count))

    client_sock.close()


if __name__ == '__main__':
    main()


