# Project-2: Threading and Multi-Core Applications


## "Computing a Mean"


![Threaded Mean Speedup Graph](threaded_speedup.png)

#### Timing Data (data.bin: 8,500,000,000 samples, mean = 52.7128)

| Threads | Time (sec) | Speedup |
|---------|------------|---------|
| 1       | 11.16      | 1.00    |
| 2       | 5.92       | 1.88    |
| 4       | 3.23       | 3.45    |
| 8       | 1.97       | 5.66    |
| 16      | 1.31       | 8.52    |
| 32      | 1.21       | 9.22    |
| 36      | 1.24       | 9.00    |
| 48      | 1.22       | 9.15    |
| 64      | 1.25       | 8.93    |
| 72      | 1.26       | 8.86    |
| 84      | 1.27       | 8.79    |

---

### Q: For the graph, note the shape of the curve. Does it "converge" to some general value? What's the maximum speedup you got from threading? What happens when you use more cores than are available in the hardware?

**Does it converge?** Yes the curve appears to converge around 9.X after about 16-32 threads.

**Maximum speedup:** 9.22× at 32 threads.

**What happens past hardware cores (72)?** When using 84 threads (exceeding the 72 hardware threads), tthe speedup actually decreases by a bit to 8.79x. I think this is due to thread overhead and there only being 72 cores artificially making 84 which just adds overhead with no actual benefits

---

### Q: Considering the number of cores in the system, do you get a linear scaling of performance as you add more cores?

No adding more cores does not provide linear scaling.

The scaling starts almost linear but eventually quickly levels out :
- 2 threads: 1.88× (94% of ideal)
- 4 threads: 3.45× (86% of ideal)
- 8 threads: 5.66× (71% of ideal)
- 16 threads: 8.52× (53% of ideal)
- 32+ threads: ~9.2× (29% of ideal)

This sub-linear scaling is happening the mean computation is bounded by memory which means the threads are spending most of their time waiting for data from RAM and not actually computing. Which places the bottleneck at the memory bandwidth.

---

### Q: Looking at your graph, what value would you propose for P, and describe how you arrived at that value.

Using Amdahl's Law: `Speedup_max = 1 / (1 - P)`

Since the speedup plateaus at approximately 9.2×:

```
9.2 = 1 / (1 - P)
1 - P = 1/9.2 ≈ 0.109
P ≈ 0.89
```

**So P ≈ 0.89 (89%)**

So this means that 89% of the program is able to be parallelized while 11% reamins serial and came to this conclusion by observing where the speedup curve plateaus and applying the inverse of Amdahl's formula.

---

### Q: Finally, consider the kernel of the mean computation. How many bytes of data are required per iteration? What's the associated bandwidth used by the kernel? Is that value consistent when you consider threaded versions?

**Bytes per iteration:** 4 bytes (one `float` value per iteration)

**Total data:** 8,500,000,000 × 4 bytes = 34 GB

**Bandwidth calculations:**

| Threads | Time (sec) | Bandwidth (GB/s) |
|---------|------------|------------------|
| 1       | 11.16      | 3.05             |
| 8       | 1.97       | 17.3             |
| 32      | 1.21       | 28.1             |

**Is it consistent?** Yes, the bandwidth scales in direct proportion to the speedups data I got. The ratio of the peak bandwidth (28.1 GB/s) to the single-thread bandwidth (3.05 GB/s) is about 9.2 which closely matches the maximum speedup achieved. The alignment confirms that the speedup is being throttled by the memory. The plateau at around 28 GB/s reflects the system's memory bandwidth limit. 

---

## "Computing a Volume" Questions

### Timing Data and Speedup Graph

![SDF Volume Speedup Graph](sdf_speedup.png)

#### Timing Data (1,000,000,000 samples, volume ≈ 0.4764)

| Threads | Time (sec) | Speedup |
|---------|------------|---------|
| 1       | 22.82      | 1.00    |
| 2       | 11.44      | 1.99    |
| 4       | 5.98       | 3.82    |
| 8       | 3.09       | 7.39    |
| 16      | 1.78       | 12.82   |
| 32      | 1.02       | 22.37   |
| 64      | 0.96       | 23.77   |
| 72      | 0.89       | 25.64   |

---

### Q: Do you get similar performance curve to `threaded.out`?

**No, the performance curves are significantly different.**

| Metric | threaded.out | sdf.out |
|--------|--------------|---------|
| Maximum Speedup | 9.2× | 25.6× |
| Plateau Point | ~32 threads | ~72 threads |
| Scaling Behavior | Early plateau | Near-linear longer |

**Why the difference?**

- **threaded.out is memory-bound:** Each iteration reads 4 bytes and does only a single addition. Threads have to wait on memory so memory bandwidth is the bottleneck.

- **sdf.out is compute-bound:** Each iteration generates random numbers and performs a lot of floating-point operations. Threads stay active and busy computing so the CPU cores are the bottleneck.

The Monte Carlo volume computation scales much better because it's "embarrassingly parallel" where each sample is independent and requires no shared memory access during computation.