# SSU-CS-351
Computer architecture 
* Project 1



## Analysis and reporting:

1) Which program is fastest? Is it always the fastest?
* The fastest program is new.cpp when the program is optimized. It is not always the fastest because with size grows hashing is more optimal. <br>

2) Which program is slowest? Is it always the slowest?

* The slowest program from my testing was list.cpp and it is the slowest due to lists nod. From my experiemnt yes.

3) Was there a trend in program execution time based on the size of data in each Node? If so, what, and why?

* Yes there was a trend where when the size of the data in each node increased the total runtime increased for all programs. I believe this happens because each node contains more bytes that have to be initialized and hased so as the data blocks got larger it took more time

4) Was there a trend in program execution time based on the length of the block chain?

* Increasing the number of block led to longer runtimes at a linear rate. 

5) Consider heap breaks, what's noticeable? Does increasing the stack size affect the heap? Speculate on any similarities and differences in programs?

* While tracking heap breaks I noticed that malloc.cpp, list.cpp, and new.cpp all made multiple calls to expand the heap as more nodes were allocated. Once enough memory pages were reserved, heap expansions diminished.

6) Considering either the malloc.cpp or alloca.cpp versions of the program, generate a diagram showing two Nodes. Include in the diagram


head → [ Node A ] tail → [ Node B ] → null
          |             |
          | bytes*      | bytes*
          ↓             ↓
       [6 bytes]     [6 bytes]

7) There's an overhead to allocating memory, initializing it, and eventually processing (in our case, hashing it). For each program, were any of these tasks the same? Which one(s) were different?

* All the programs allocated memory by initlaizing them with random bytes and hashing it but the way each one allocated memory was different. list used std::list, new used C++'s new operator, malloc used malloc(), and alloca used alloca().

8) As the size of data in a Node increases, does the significance of allocating the node increase or decrease?

* As the Node's data size increases the cost of allocation decreases.

