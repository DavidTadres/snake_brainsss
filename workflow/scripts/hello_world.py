def print_hi(logfile, args, arg2,
             ):

    #printlog = getattr(brainsss.Printlog(logfile=logfile), 'print_to_log')
    #sys.stderr = brainsss.logger_stderr(logfile)  # print errors to log file

    #sys.stdout = brainsss.LoggerStdout(logfile)
    print('Hi ' + args)  # Press ⌘F8 to toggle the breakpoint.
    print(arg2)
    #print(pathlib.Path(__file__).parent.resolve())

    #printlog('test')
    #print(error)
    #logger.debug('your message')

def print_bye(logfile, args):
    print(args)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi("Pycharm")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
