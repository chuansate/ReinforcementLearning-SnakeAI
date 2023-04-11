"""
a = 0

def increment(num):
    num += 100

increment(a)
print("a = ", a)
"""
import numpy as np

"""class Test:
    def __init__(self):
        self.att = 0


def increment(object):
    object.att += 100

def increment_again(object):
    object.att += 10000

def main():
    t = Test()
    increment(t) # 100
    print(t.att)
    increment_again(t)  # 10100
    print(t.att)

main()"""

# Hypothesis:
# Immutable objects such as float, int, string will be passed as reference to a function
# Mutable objects such as list, dictionary, and user-defined objects will be passed as pointers to a function
li = [3, 2, 1]
li.pop()
print(li)