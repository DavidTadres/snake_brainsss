import sys
import pathlib

def print_hi(args, arg2):
    print('Hi ' + args)  # Press ⌘F8 to toggle the breakpoint.
    print(arg2)
    print(pathlib.Path(__file__).parent.resolve())

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi("Pycharm")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
