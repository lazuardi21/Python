import time
from math import floor, log10


def get_digit(num):
    if num < 10:
        print(num)
    else:
        get_digit(num // 10)
        print(num % 10)


def digits(n):
    """generator which returns digits in left to right order"""

    k = floor(log10(n))
    for e in range(k, -1, -1):
        d, n = divmod(n, 10**e)
        yield d


def run_once(x, _has_run=[]):
    if _has_run:
        return
    # print("run_once doing stuff")

    _has_run.append(1)
    # print(x)

    a = list(digits(x))
    print(a)
    return x


x = time.time()
y = time.time_ns()
print(y)

# def my_function2(_has_run=[]):
#     if _has_run:
#         return
#     print("my_function2 doing some other stuff")
#     _has_run.append(1)


a = run_once(x)

for i in range(10000):
    run_once(x)

    if i == 9999:
        endtime = time.time_ns()
        inv = (endtime - y)/(10**12)

        print(endtime)
        s = run_once(x)
        print(inv)
    # my_function2()
