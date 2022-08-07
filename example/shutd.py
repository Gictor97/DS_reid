def run_sh():

    import os
    os.system('shutdown')  #自动关机

if __name__ == '__main__':
    import time
    print('服务器关机')
    time.sleep(10)
    run_sh()
