import os
import pickle


def SimpleCalculator(args):
    os.mkdir(args.save_dir, exist_ok=True)
    total = args.number1 + args.number2
    with open(f"{args.save_dir}/total.pickle", "wb") as f:
        pickle.dump(total, f)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download.')
    parser.add_argument('--save_dir', default='data', type=str, help='data directory')
    parser.add_argument('--number1', default=20, type=int, help='data directory')
    parser.add_argument('--number2', default=50, type=int, help='data directory')

    args = parser.parse_args()
    SimpleCalculator(args)
