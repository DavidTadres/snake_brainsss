from scripts import hello_world

# Can't put a rule into a loop!!!

rule class_test_rule:
    input:
        path_1 = '/Users/dtadres/Documents/test.txt',
        #path_2 = '/Users/dtadres/Documents/test2.txt'
    shell: "python3 scripts/hello_world.py --input {input} --ouput 'help.txt'"

"""for i in range(3):

    rule HelloSnake:
        #shell:
        #    'python3 hello_world.py $args'
        threads: 2
        output: filename="HelloSnake{wild}.txt"
        run:
            print('\nExecuting HelloSnake rule\n')

            hello_world.print_hi(logfile='foo',
                args='world',
                arg2='i'
            )"""