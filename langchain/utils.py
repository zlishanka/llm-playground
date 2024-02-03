import sys
from datetime import datetime

def print_info(msg):
    print(str(datetime.now()) + ' [Info] ' + msg)
    sys.stdout.flush()

def print_warning(msg):
    print(str(datetime.now()) + ' [Warning] ' + msg)
    sys.stdout.flush()

def print_error(msg):
    print(str(datetime.now()) + ' [Error] ' + msg)
    sys.stdout.flush()