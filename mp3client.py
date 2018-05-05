
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

    sock = socket.socket()
    sock.bind(this_machine)
    sock.connect(server)

    with open('warofchange.mp4', 'rb') as vid:
        frame = True; count = 0; buf_size = 1024
        while frame:
            frame = vid.read(buf_size) # read and send 1024 bytes
            sock.send(frame)
            count += 1
            log = 'log: sent {} kb to {}'.format(count, server[0])
            print('\r'*len(log)+log, end='')  # pretty printing!
        sock.close()
        print()


if __name__ == '__main__':
    main()


