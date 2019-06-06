# Uses python3
import sys
# from tqdm import tqdm


def optimal_weight(W, w):
    # write your code here

    mem = [None for _ in range(W)]

    for i in w:
        'save the intermediate results into mem'




    result = 0
    for x in w:
        if result + x <= W:
            result = result + x
    return result

if __name__ == '__main__':
    input = sys.stdin.read()
    W, n, *w = list(map(int, input.split()))
    print(optimal_weight(W, w))
