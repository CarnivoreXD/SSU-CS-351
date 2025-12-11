#!/usr/bin/env python3
"""
Generate speedup graphs for Project-2 README
Run: python3 make_graphs.py
"""

import matplotlib.pyplot as plt

# =============================================================================
# DATA FROM YOUR TIMING RUNS
# =============================================================================

# threaded.out data
threaded_threads = [1, 2, 4, 8, 16, 32, 36, 48, 64, 72, 84]
threaded_speedup = [1.00, 1.88, 3.45, 5.66, 8.52, 9.22, 9.00, 9.15, 8.93, 8.86, 8.79]

# sdf.out data
sdf_threads = [1, 2, 4, 8, 16, 32, 64, 72]
sdf_speedup = [1.00, 1.99, 3.82, 7.39, 12.82, 22.37, 23.77, 25.64]

# =============================================================================
# GRAPH 1: threaded.out speedup
# =============================================================================

plt.figure(figsize=(10, 6))
plt.plot(threaded_threads, threaded_speedup, 'b-o', linewidth=2, markersize=8, label='Actual Speedup')

# Add ideal linear speedup line for reference
max_threads = max(threaded_threads)
plt.plot([1, max_threads], [1, max_threads], 'r--', alpha=0.5, label='Ideal Linear Speedup')

# Add horizontal line at max speedup
max_speedup = max(threaded_speedup)
plt.axhline(y=max_speedup, color='g', linestyle=':', alpha=0.5, label=f'Max Speedup ({max_speedup}x)')

# Mark the 72-core hardware limit
plt.axvline(x=72, color='orange', linestyle='--', alpha=0.5, label='Hardware Threads (72)')

plt.xlabel('Number of Threads', fontsize=12)
plt.ylabel('Speedup', fontsize=12)
plt.title('threaded.out: Mean Computation Speedup\n(8.5 billion samples, data.bin)', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.xlim(0, 90)
plt.ylim(0, 15)

plt.tight_layout()
plt.savefig('threaded_speedup.png', dpi=150)
print("Created: threaded_speedup.png")

# =============================================================================
# GRAPH 2: sdf.out speedup
# =============================================================================

plt.figure(figsize=(10, 6))
plt.plot(sdf_threads, sdf_speedup, 'b-o', linewidth=2, markersize=8, label='Actual Speedup')

# Add ideal linear speedup line for reference
max_threads = max(sdf_threads)
plt.plot([1, max_threads], [1, max_threads], 'r--', alpha=0.5, label='Ideal Linear Speedup')

# Add horizontal line at max speedup
max_speedup = max(sdf_speedup)
plt.axhline(y=max_speedup, color='g', linestyle=':', alpha=0.5, label=f'Max Speedup ({max_speedup}x)')

# Mark the 72-core hardware limit
plt.axvline(x=72, color='orange', linestyle='--', alpha=0.5, label='Hardware Threads (72)')

plt.xlabel('Number of Threads', fontsize=12)
plt.ylabel('Speedup', fontsize=12)
plt.title('sdf.out: Monte Carlo Volume Speedup\n(1 billion samples)', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.xlim(0, 80)
plt.ylim(0, 30)

plt.tight_layout()
plt.savefig('sdf_speedup.png', dpi=150)
print("Created: sdf_speedup.png")

# =============================================================================
# GRAPH 3: Comparison of both (BONUS)
# =============================================================================

plt.figure(figsize=(10, 6))
plt.plot(threaded_threads, threaded_speedup, 'b-o', linewidth=2, markersize=8, label='threaded.out (Memory-bound)')
plt.plot(sdf_threads, sdf_speedup, 'g-s', linewidth=2, markersize=8, label='sdf.out (Compute-bound)')

# Add ideal linear speedup line
plt.plot([1, 72], [1, 72], 'r--', alpha=0.5, label='Ideal Linear Speedup')

# Mark the 72-core hardware limit
plt.axvline(x=72, color='orange', linestyle='--', alpha=0.5, label='Hardware Threads (72)')

plt.xlabel('Number of Threads', fontsize=12)
plt.ylabel('Speedup', fontsize=12)
plt.title('Speedup Comparison: Memory-bound vs Compute-bound', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.xlim(0, 90)
plt.ylim(0, 30)

plt.tight_layout()
plt.savefig('comparison_speedup.png', dpi=150)
print("Created: comparison_speedup.png")

print("\nDone! Add these to your README:")
print("  ![Threaded Mean Speedup Graph](threaded_speedup.png)")
print("  ![SDF Volume Speedup Graph](sdf_speedup.png)")
print("  ![Comparison](comparison_speedup.png)")
