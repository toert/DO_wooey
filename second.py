import argparse
import sys

parser = argparse.ArgumentParser(description="Find the sum of all the numbers below a certain number.")
parser.add_argument('--m', help='Количество валют в портфеле', type=int, default=0)
parser.add_argument('--pairs', action='append', help='Пары валют, через запятую', choices=['rock', 'paper', 'scissors'])
parser.add_argument('--n', help='Количество исторических периодов', type=int)
parser.add_argument('--T', help='Длина 1 периода в минутах', type=int)
parser.add_argument('--P', help='Предел уменьшения размера', type=int, default=100)

def main():
    args = parser.parse_args()
    print(args)
    return 0

if __name__ == "__main__":
    sys.exit(main())