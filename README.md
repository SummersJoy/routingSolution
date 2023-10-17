# An efficient array oriented implementation of memetic algorithm to solve variants of VRPs

Memetic algorithm is basically a genetic algorithm enhanced by local search.

The choice of data structures to represent the solution of VRPs can significantly affect the run time. A few points
must be considered below:

* insertion operation
* deletion operation
* query the location of a certain customer

## Array Oriented Route Representation

Consider the following trip represented via lists:

[[0,1,2,0], [0,3,4,5,0]]

The task of querying the location of each customer in the trip can be achieved in O(1) time by maintaining a lookup
table:

| index | route index | position index | 
|-------|:-----------:|:--------------:|
| 0     |     -1      |       -1       |
| 1     |      0      |       0        |
| 2     |      0      |       1        |
| 3     |      1      |       0        |
| 4     |      1      |       1        |
| 5     |      1      |       2        |

There are two routes in the trip: [0,1,2,0] and [0,3,4,5,0]. The depot (index 0) by default is set to belong to
route -1 and position -1. Customer 1 is the first customer in the first trip, so that second row is 1, 0, 0.

Without the lookup table, it will take O(n) time to search for the location of each customer.

The previous and next customer of a given customer is kept in lookup_prev and lookup_next. For example, the previous
customer of 4 is 3 and the next customer of 4 is 5.

The lookup_prev is:

| index | value |
|-------|-------|
| 0     | -1    |
| 1     | 0     |
| 2     | 1     |
| 3     | 0     |
| 4     | 3     |
| 5     | 4     |

The lookup_next is

| index | value |
|-------|-------|
| 0     | -1    |
| 1     | 2     |
| 2     | 0     |
| 3     | 4     |
| 4     | 5     |
| 5     | 0     |
![](./assets/fig/relocate.jpg)