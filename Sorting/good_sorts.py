"""
This file corresponds to the first graded lab of 2XC3.
Feel free to modify and/or add functions to this file.

In contains traditional implementations for:
1) Quick sort
2) Merge sort
3) Heap sort

Author: Vincent Maccio
"""
import timeit
import random
import matplotlib.pyplot as plot

# ************ Quick Sort ************
def quicksort(L):
    copy = quicksort_copy(L)
    for i in range(len(L)):
        L[i] = copy[i]


def quicksort_copy(L):
    if len(L) < 2:
        return L
    pivot = L[0]
    left, right = [], []
    for num in L[1:]:
        if num < pivot:
            left.append(num)
        else:
            right.append(num)
    return quicksort_copy(left) + [pivot] + quicksort_copy(right)

def dual_quicksort(L):
    if len(L) < 2:
        return L
    if L[0]>L[1]:
        temp = L[0]
        L[0], L[1] = L[1] , temp
    pivot1 = L[0]
    pivot2 = L[1]

    left, mid, right = [], [], []
    for num in L[2:]:
        if num < pivot1 and num < pivot2:
            left.append(num)
        elif num > pivot1 and num > pivot2:
            right.append(num)
        else:
            mid.append(num)
    return dual_quicksort(left) + [pivot1] + dual_quicksort(mid) + [pivot2] + dual_quicksort(right)


# *************************************


# ************ Merge Sort *************

def mergesort(L):
    if len(L) <= 1:
        return
    mid = len(L) // 2
    left, right = L[:mid], L[mid:]

    mergesort(left)
    mergesort(right)
    temp = merge(left, right)

    for i in range(len(temp)):
        L[i] = temp[i]

def bottom_up_mergesort(L):
    window_size = 1
    while window_size < len(L):
        # print("Window size:", window_size)
        for i in range(0, len(L), 2 * window_size):
            left = L[i: i + window_size]
            right = L[i + window_size: (i + 2 * window_size)]
            temp = merge(left, right)
            L[i:i + (2 * window_size)] = temp
        #   print("Merged:", left, "and", right, "=>", temp)


        window_size *= 2


def merge(left, right):
    L = []
    i = j = 0

    while i < len(left) or j < len(right):
        if i >= len(left):
            L.append(right[j])
            j += 1
        elif j >= len(right):
            L.append(left[i])
            i += 1
        else:
            if left[i] <= right[j]:
                L.append(left[i])
                i += 1
            else:
                L.append(right[j])
                j += 1
    return L

# *************************************

# ************* Heap Sort *************

def heapsort(L):
    heap = Heap(L)
    for _ in range(len(L)):
        heap.extract_max()

class Heap:
    length = 0
    data = []

    def __init__(self, L):
        self.data = L
        self.length = len(L)
        self.build_heap()

    def build_heap(self):
        for i in range(self.length // 2 - 1, -1, -1):
            self.heapify(i)

    def heapify(self, i):
        largest_known = i
        if self.left(i) < self.length and self.data[self.left(i)] > self.data[i]:
            largest_known = self.left(i)
        if self.right(i) < self.length and self.data[self.right(i)] > self.data[largest_known]:
            largest_known = self.right(i)
        if largest_known != i:
            self.data[i], self.data[largest_known] = self.data[largest_known], self.data[i]
            self.heapify(largest_known)

    def insert(self, value):
        if len(self.data) == self.length:
            self.data.append(value)
        else:
            self.data[self.length] = value
        self.length += 1
        self.bubble_up(self.length - 1)

    def insert_values(self, L):
        for num in L:
            self.insert(num)

    def bubble_up(self, i):
        while i > 0 and self.data[i] > self.data[self.parent(i)]:
            self.data[i], self.data[self.parent(i)] = self.data[self.parent(i)], self.data[i]
            i = self.parent(i)

    def extract_max(self):
        self.data[0], self.data[self.length - 1] = self.data[self.length - 1], self.data[0]
        max_value = self.data[self.length - 1]
        self.length -= 1
        self.heapify(0)
        return max_value

    def left(self, i):
        return 2 * (i + 1) - 1

    def right(self, i):
        return 2 * (i + 1)

    def parent(self, i):
        return (i + 1) // 2 - 1

    def __str__(self):
        height = math.ceil(math.log(self.length + 1, 2))
        whitespace = 2 ** height
        s = ""
        for i in range(height):
            for j in range(2 ** i - 1, min(2 ** (i + 1) - 1, self.length)):
                s += " " * whitespace
                s += str(self.data[j]) + " "
            s += "\n"
            whitespace = whitespace // 2
        return s

# *************************************

def create_random_list(length, max_value):
    return [random.randint(0, max_value) for _ in range(length)]

def create_near_sorted_list(length, max_value, swaps):
    L = create_random_list(length, max_value)
    L.sort()
    for _ in range(swaps):
        r1 = random.randint(0, length - 1)
        r2 = random.randint(0, length - 1)
        swap(L, r1, r2)
    return L
# I have created this function to make the sorting algorithm code read easier
def swap(L, i, j):
    L[i], L[j] = L[j], L[i]

def experiment4(n):
    times1=[]
    times2=[]
    times3=[]
    for i in range(n):
        list = create_random_list(i, 500)
        start = timeit.default_timer()
        quicksort(list)
        end = timeit.default_timer()
        times1.append(end - start)
    plot.plot(times1, label = "quicksort")
    for i in range(n):
        list = create_random_list(i, 500)
        start = timeit.default_timer()
        mergesort(list)
        end = timeit.default_timer()
        times2.append(end - start)
    plot.plot(times2, label = "mergesort")
    for i in range(n):
        list = create_random_list(i, 500)
        start = timeit.default_timer()
        heapsort(list)
        end = timeit.default_timer()
        times3.append(end - start)
    plot.plot(times3, label = "heapsort")
    plot.legend()
    plot.title("List Length v. Time")
    plot.xlabel("List Length")
    plot.ylabel("Time (seconds)")
    plot.show()

def experiment5(n):
    length = 100
    times1 = []
    times2 = []
    times3 = []
    #num swaps before quicksort is appealing..
    for i in range(n):
        list = create_near_sorted_list(length,500,i)
        start = timeit.default_timer()
        quicksort(list)
        end = timeit.default_timer()
        times1.append(end - start)
    plot.plot(times1, label = 'quicksort')
    for i in range(n):
        list = create_near_sorted_list(length,500,i)
        start = timeit.default_timer()
        mergesort(list)
        end = timeit.default_timer()
        times2.append(end - start)
    plot.plot(times2, label = "mergesort")
    for i in range(n):
        list = create_near_sorted_list(length,500,i)
        start = timeit.default_timer()
        heapsort(list)
        end = timeit.default_timer()
        times3.append(end - start)
    plot.plot(times3, label = "heapsort")
    plot.legend()
    plot.title("Swaps v. Time")
    plot.xlabel("Swaps")
    plot.ylabel("Time (seconds)")
    plot.show()

def experiment6(n):
    times1=[]
    times2=[]

    for i in range(n):
        list = create_random_list(i, 500)
        start = timeit.default_timer()
        quicksort(list)
        end = timeit.default_timer()
        times1.append(end - start)
    plot.plot(times1, label = "quicksort")

    for i in range(1,n):
        list = create_random_list(i, 500)
        start = timeit.default_timer()
        list = dual_quicksort(list)
        end = timeit.default_timer()
        times2.append(end - start)
    plot.plot(times2, label = "dual quicksort")
    plot.legend()
    plot.title("List Length v. Time")
    plot.xlabel("List Length")
    plot.ylabel("Time (seconds)")
    plot.show()


def experiment7(n):
    times1 = []  # Traditional Merge Sort
    times2 = []
    for i in range(n):
        list = create_random_list(i, i)

        start = timeit.default_timer()
        mergesort(list)
        end = timeit.default_timer()
        times1.append(end - start)
    plot.plot(times1, label="Traditional Merge Sort")

    for i in range(n):
        list = create_random_list(i, i)

        start = timeit.default_timer()
        bottom_up_mergesort(list)
        end = timeit.default_timer()
        times2.append(end - start)

    plot.plot(times2, label="Bottom-up Merge Sort")
    plot.legend()
    plot.title("List Length vs. Time")
    plot.xlabel("List Length")
    plot.ylabel("Time (seconds)")
    plot.show()

# ******************* Insertion sort code *******************

# This is the traditional implementation of Insertion Sort.
def swap(L, i, j):
    L[i], L[j] = L[j], L[i]

def insertion_sort(L):
    for i in range(1, len(L)):
        insert(L, i)


def insert(L, i):
    while i > 0:
        if L[i] < L[i-1]:
            swap(L, i-1, i)
            i -= 1
        else:
            return

def experiment8(n):
    times1 = []
    times2 = []
    times3 = []

    for i in range(n):
        list = create_random_list(i,i)

        start = timeit.default_timer()
        insertion_sort(list)
        end = timeit.default_timer()
        times1.append(end - start)

    for i in range(n):
        list = create_random_list(i,i)

        start = timeit.default_timer()
        mergesort(list)
        end = timeit.default_timer()
        times2.append(end - start)

    for i in range(n):
        list = create_random_list(i,i)

        start = timeit.default_timer()
        quicksort(list)
        end = timeit.default_timer()
        times3.append(end - start)

    plot.plot(times1, label="Insertion Sort")
    plot.plot(times2, label="Merge Sort")
    plot.plot(times3, label="Quick Sort")

    plot.legend()
    plot.title("List Length vs. Time")
    plot.xlabel("List Length")
    plot.ylabel("Time (seconds)")

    plot.show()

experiment8(50)
