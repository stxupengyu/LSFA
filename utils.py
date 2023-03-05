import time

def time_since(start_time):
    return time.time()-start_time

def cprint(name, value):
    print('=============%s=============='%name)
    print(value)
    print('=============%s=============='%name)