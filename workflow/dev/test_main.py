
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--fly_directory", help="Folder of fly to save log")
    parser.add_argument("--test", help="Folder of fly to save log")
    args = parser.parse_args()

    print(args.fly_directory)

    if hasattr(args, 'fly_directory'):
        print(args.fly_directory)
    else:
        print('didnt work')