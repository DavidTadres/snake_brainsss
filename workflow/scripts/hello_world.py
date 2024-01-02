import sys

def print_hi(
    logfile,
    args,
    arg2,
):
    # printlog = getattr(brainsss.Printlog(logfile=logfile), 'print_to_log')
    # sys.stderr = brainsss.logger_stderr(logfile)  # print errors to log file

    # sys.stdout = brainsss.LoggerStdout(logfile)
    print("Hi " + args)  # Press ⌘F8 to toggle the breakpoint.
    print(arg2)
    # print(pathlib.Path(__file__).parent.resolve())

    # printlog('test')
    # print(error)
    # logger.debug('your message')


def print_bye(logfile, args):
    print(args)

class ShellTest():
    def __init__(self, args):
        print("args: " + repr(args))
        self.shell_test()
    def shell_test(self):
        print('shell test')


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = sys.argv[1]
    print(sys.argv[-1])
    ShellTest(args)
    #print_hi("Pycharm")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
