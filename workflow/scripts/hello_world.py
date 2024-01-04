import sys
import argparse
import time
import multiprocessing
import itertools
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

def parallel_function(index, foo):
    print(repr(index) + foo)
    time.sleep(index)



# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    index = [0,1,2,3,4,5,6,7,8,9,10,11]
    cores = 4
    #with multiprocessing.Pool(cores) as p:
    #    p.map(parallel_function, index)

    foo = 'bar'
    # Create pool
    pool = multiprocessing.Pool(cores)
    # Keep track of number of processes
    running_processes = 0
    counter = 0
    # Make a list to keep track of the spawned child processes
    child_processes = []
    # Define what the max number of processes is
    max_processes = cores
    # Loop through index, yield 0, 1 etc.
    for i in index:
        # Run until break
        while True:
            # Only fork a new process is there are less processes running than max_processes
            if running_processes < max_processes:
                # Define process: Target is a function, args are the arguments
                p = multiprocessing.Process(target = parallel_function, args = (i,foo))
                # start process
                p.start()
                # To keep track of child processes, add to list
                child_processes.append(p)
                # to keep track of running_processes
                running_processes +=1
                counter +=1
                # get out of the current 'while' loop and go back to the for loop
                break
            # Processor wait loop if we don't have running_processes < max_processes
            else:
                # Stay here until break is called
                while True:
                    # loop through the child_processes
                    for current_child_process in range(len(child_processes)):
                        # Check if process is still running
                        if child_processes[current_child_process].is_alive():
                            # Continue for loop (i.e. check next child_process)
                            continue
                        else:
                            # If it's found that a child process isn't running anymore,
                            # remove the item at the current index
                            child_processes.pop(current_child_process)
                            # Subtract running processes by one
                            running_processes -= 1
                            # and break the for loop
                            break
                    # We are here either because the for loop finished or becuse
                    # it was found that a process is not running anymore.
                    # Check if we have less running processes than max processes
                    if running_processes < max_processes:
                        # If yes, break this inner while loop and go back to the
                        # outer while loop that keeps to start a new child process
                        break
                        # Else stay in this while loop and check again for processes
                        # that are finished.

    # wait for remaining processes to complete --> this is the same code as the
    # processor wait loop above
    while len(child_processes) > 0:
        for next in range(len(child_processes)):
            if child_processes[next].is_alive():
                continue
            else:
                child_processes.pop(next)
                running_processes -= 1
                break  # need to break out of the for-loop,
                # because the child_list index is changed by pop





    #parser = argparse.ArgumentParser()
    #parser.add_argument("--input", help="use this as input")
    #parser.add_argument("--ouput", help="use this as output")
    #args = parser.parse_args()

    #print(f"Args: {args}\nCommand Line: {sys.argv}\ninput: {args.input}")
    #print(sys.argv)
    #args = sys.argv[1]
    #print(sys.argv[-1])
    #ShellTest(args)
    #print_hi("Pycharm")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
