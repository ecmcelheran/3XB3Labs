import random
import matplotlib.pyplot as plot
import timeit


# Part 1 - Implementations
def ks_brute_force(items, capacity):
    powerset = power_set(items)
    value = 0
    # create all possible subsets to find optimal solution
    for s in powerset:
        x = sum(s[i][1] for i in range(len(s)))
        if x <= capacity:
            v = sum(s[i][0] for i in range(len(s)))
            if v > value:
                value = v
    return value


# Additional functions for brute force method
def power_set(set):
    if not set:
        return [[]]
    return power_set(set[1:]) + add_to_each(power_set(set[1:]), set[0])


def add_to_each(sets, element):
    copy = sets.copy()
    for set in copy:
        set.append(element)
    return copy


def ks_rec(items, capacity):
    if capacity == 0 or len(items) == 0:
        return 0
    if items[len(items)-1][1] > capacity:
        return ks_rec(items[:-1], capacity)
    else:
        return max(items[len(items)-1][0] + ks_rec(items[:-1], capacity-items[len(items)-1][1]), ks_rec(items[:-1], capacity))


def ks_bottom_up(items, capacity):
    bu = [[0 for i in range(capacity + 1)] for j in range(len(items) + 1)]

    for i in range(1, len(items) + 1):
        for j in range(1, capacity + 1):
            weight = items[i - 1][1]
            value = items[i - 1][0]
            # if weight > j:
            #    bu[i][j] = bu[i - 1][j]
            if weight <= j:
                bu[i][j] = max(bu[i - 1][j], value + bu[i - 1][j - weight])
            else:
                bu[i][j] = bu[i - 1][j]
        #     bu[i][j] = max(bu[i - 1][j], value + bu[i - 1][j - weight])

    return bu[len(items)][capacity]


def ks_top_down(items, capacity):
    td = {}
    #    td = [[0 for i in range(capacity + 1)] for j in range(len(items) + 1)]
    for i in range(capacity + 1):
        td[(0, i)] = 0
    for j in range(len(items) + 1):
        td[(j, 0)] = 0
    ks_top_down_aux(items, len(items), capacity, td)
    return td[(len(items), capacity)]
def ks_top_down_aux(items, i, j, td):
    if (i, j) in td:
        return td[(i, j)]

    if items[i - 1][1] > j:
        td[(i, j)] = ks_top_down_aux(items, i - 1, j, td)
    else:
        value1 = items[i - 1][0] + ks_top_down_aux(items, i - 1, j - items[i - 1][1], td)
        value2 = ks_top_down_aux(items, i - 1, j, td)
        td[(i, j)] = max(value1, value2)

    return td[(i, j)]


def create_rand_item_set(length, minl, maxl, minw, maxw):
    items = []
    for i in range(length):
        items.append([random.randint(minl, maxl), random.randint(minw, maxw)])
    return items


# runs in the worst case
def num_of_wc_runs(n, m):
    global runs
    runs = {}

    for i in range(1, n+1):
        runs[(i, 1)] = i

    for i in range(1, m+1):
        runs[(1, i)] = 1
        runs[(0, i)] = 0

    for i in range(2, n+1):
        for j in range(2, m+1):
            runs[(i, j)] = 9999999
            k = next_setting(i, j)
            runs[(i, j)] = 1 + max(runs[(k-1, j-1)], runs[(i-k, j)])
    return runs[(n, m)]

# next setting
def next_setting(n, m):
    # minimize runs
    minrun = 9999999999
    k = 0
    for i in range(2, n+1):
        t = 1 + max(runs[(i-1, m-1)], runs[(n-i, m)])
        if t < minrun:
            minrun = t
            k = i
    return k


def implementation_bruteforce_rec(n):
    times1 = []
    times2 = []
    for i in range(n):
        my_set = create_rand_item_set(n, 1, 15, 1, 10)
        start = timeit.default_timer()
        ks_brute_force(my_set, 12)
        end = timeit.default_timer()
        times1.append(end - start)
        start = timeit.default_timer()
        ks_rec(my_set, 12)
        end = timeit.default_timer()
        times2.append(end - start)

    plot.plot(times1, label="ks_brute_force")
    plot.plot(times2, label="ks_rec")
    plot.legend()
    plot.title("Set Length vs. Time")
    plot.xlabel("Set Length")
    plot.ylabel("Time (seconds)")
    plot.show()

def implementation_td_bu(n):
    times1 = []
    times2 = []
    w1 = 10
    w2 = 500
    for i in range(n):
        my_set = create_rand_item_set(n, 1, 15, 1, w1)

        start = timeit.default_timer()
        ks_top_down(my_set, 12)
        end = timeit.default_timer()
        times1.append(end - start)
        start = timeit.default_timer()
        ks_bottom_up(my_set, 12)
        end = timeit.default_timer()
        times2.append(end - start)

    plot.plot(times1, label="ks_top_down")
    plot.plot(times2, label="ks_bottom_up")

    plot.legend()
    plot.title("Set Length vs. Time")
    plot.xlabel("Set Length")
    plot.ylabel("Time (seconds)")

    plot.show()

