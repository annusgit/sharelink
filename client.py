
"""
    Author: Annus Zulfiqar
    Date: 5th May, 2018
"""

from __future__ import print_function
from __future__ import division
import socket

server = ('127.0.0.1', 50000)
this_machine = ('127.0.0.1', 50001)


def main():

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(this_machine)
    sock.connect(server)

    with open('received.zip', 'w') as received:
        # let's receive this zip file
        buf_size = 1024; count = 0
        print('log: receiving now...')
        while True:
            data = sock.recv(buf_size); count += 1
            if not data:
                break
            received.write(data)
            pass

        print('log: received {} kbs to the server'.format(count))


    pass


if __name__ == '__main__':
    main()










