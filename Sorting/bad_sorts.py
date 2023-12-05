"""
This file corresponds to the first graded lab of 2XC3.
Feel free to modify and/or add functions to this file.
"""
import timeit
import random
import matplotlib.pyplot as plot


# Create a random list length "length" containing whole numbers between 0 and max_value inclusive
def create_random_list(length, max_value):
    return [random.randint(0, max_value) for _ in range(length)]

# Creates a near sorted list by creating a random list, sorting it, then doing a random number of swaps
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

# ******************* Insertion sort code *******************

# This is the traditional implementation of Insertion Sort.
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

#Insertion Sort 2
def insertion_sort2(L):
    for i in range(len(L)):
        current = L[i]
        j=i
        while j>0 and L[j-1]>current:
            L[j] = L[j-1]
            j-=1
        L[j]=current

# ******************* Bubble sort code *******************

# Traditional Bubble sort
def bubble_sort(L):
    for i in range(len(L)):
        for j in range(len(L) - 1):
            if L[j] > L[j+1]:
                swap(L, j, j+1)

#Bubble sort 2
def bubble_sort2(L):
    for i in range(len(L)):
        for j in range(len(L) - 1):
            if L[j] > L[j+1]:
                current = L[j+1]
                a=j+1
                while a>0:
                    if current<L[a-1]:
                        L[a]=L[a-1]
                        a-=1
                    else:
                        break
                L[a]=current



# ******************* Selection sort code *******************

# Traditional Selection sort
def selection_sort(L):
    for i in range(len(L)):
        min_index = find_min_index(L, i)
        swap(L, i, min_index)


def find_min_index(L, n):
    min_index = n
    for i in range(n+1, len(L)):
        if L[i] < L[min_index]:
            min_index = i
    return min_index

# selection sort 2 code
def selection_sort2(L):
    i,j = 0, len(L)-1
    while i<j:
        index = find_min_max_index(L, i, j)
        swap(L,i,index[0])
        if index[0]!=j:
            swap(L,j,index[1])
        i+=1
        j-=1

def find_min_max_index(L,n,j):
    max_index=n
    min_index=n
    for i in range(n,j+1):
        if L[i] > L[max_index]:
            max_index = i
        if L[i] < L[min_index]:
            min_index = i
    return [max_index,min_index]


def experiment1(n):
    times1=[]
    times2=[]
    times3=[]

    for i in range(n):
        list = create_random_list(i, 500)
        start = timeit.default_timer()
        insertion_sort(list)
        end = timeit.default_timer()
        times1.append(end - start)
    plot.plot(times1, label = "insertion")
    for i in range(n):
        list = create_random_list(i, 500)
        start = timeit.default_timer()
        bubble_sort(list)
        end = timeit.default_timer()
        times2.append(end - start)
    plot.plot(times2, label = "bubble")
    for i in range(n):
        list = create_random_list(i, 500)
        start = timeit.default_timer()
        selection_sort(list)
        end = timeit.default_timer()
        times3.append(end - start)
    plot.plot(times3, label = "selection")
    plot.legend()
    plot.title("Swaps v. Time")
    plot.xlabel("Swaps")
    plot.ylabel("Time (seconds)")
    plot.show()

def experiment2(n):
    times1,times2,times3,times4,times5,times6 = [],[],[],[],[],[]

    for i in range(n):
        list = create_random_list(i,i)
        start = timeit.default_timer()
        insertion_sort(list)
        end = timeit.default_timer()
        times1.append(end - start)
    plot.plot(times1, label = "insertion")
    for i in range(n):
        list = create_random_list(i,i)
        start = timeit.default_timer()
        insertion_sort2(list)
        end = timeit.default_timer()
        times2.append(end - start)
    plot.plot(times2, label = "insertion2")

    for i in range(n):
        list = create_random_list(i,i)
        start = timeit.default_timer()
        bubble_sort(list)
        end = timeit.default_timer()
        times3.append(end - start)
    plot.plot(times3, label = "bubble")
    for i in range(n):
        list = create_random_list(i,i)
        start = timeit.default_timer()
        bubble_sort2(list)
        end = timeit.default_timer()
        times4.append(end - start)
    plot.plot(times4, label = "bubble2")

    for i in range(n):
        list = create_random_list(i,i)
        start = timeit.default_timer()
        selection_sort(list)
        end = timeit.default_timer()
        times5.append(end - start)
    plot.plot(times5, label = "selection")
    for i in range(n):
        list = create_random_list(i,i)
        start = timeit.default_timer()
        selection_sort2(list)
        end = timeit.default_timer()
        times6.append(end - start)
    plot.plot(times6, label = "selection2")
    plot.legend()
    plot.title("List Length v. Time")
    plot.xlabel("List Length")
    plot.ylabel("Time (seconds)")
    plot.show()

def experiment3(n):
    times1=[]
    times2=[]
    times3=[]
    length=100

    for i in range(n):
        list = create_near_sorted_list(length, 500, i)
        start = timeit.default_timer()
        insertion_sort(list)
        end = timeit.default_timer()
        times1.append(end - start)
    plot.plot(times1, label = "insertion")
    for i in range(n):
        list = create_near_sorted_list(length, 500, i)
        start = timeit.default_timer()
        bubble_sort(list)
        end = timeit.default_timer()
        times2.append(end - start)
    plot.plot(times2, label = "bubble")
    for i in range(n):
        list = create_near_sorted_list(length, 500, i)
        start = timeit.default_timer()
        selection_sort(list)
        end = timeit.default_timer()
        times3.append(end - start)
    plot.plot(times3, label = "selection")
    plot.legend()
    plot.title("Swaps v. Time")
    plot.xlabel("Swaps")
    plot.ylabel("Time (seconds)")
    plot.show()


experiment3(100)

