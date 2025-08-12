import time
import subprocess
import psutil
import statistics
import threading

def monitor_cpu_usage(interval, usage_log, stop_event):
    """Monitors CPU usage at fixed intervals until stop_event is set."""
    while not stop_event.is_set():
        usage_log.append(psutil.cpu_percent(interval=interval, percpu=True))

def run_and_benchmark(command, description):
    """
    Runs a Python script and records:
    - Execution time
    - CPU usage per core over time
    """
    print(f"\n=== Benchmark: {description} ===")
    usage_log = []
    stop_event = threading.Event()

    # Start CPU monitoring in a separate thread
    monitor_thread = threading.Thread(
        target=monitor_cpu_usage,
        args=(0.1, usage_log, stop_event),
        daemon=True
    )
    monitor_thread.start()

    # Time the script execution
    start_time = time.perf_counter()
    subprocess.run(["python", command], check=True)
    end_time = time.perf_counter()

    # Stop CPU monitoring
    stop_event.set()
    monitor_thread.join()

    # Report execution time
    exec_time = end_time - start_time
    print(f"Execution time: {exec_time:.3f} seconds")

    # Compute average CPU usage per core
    if usage_log:
        avg_usage_per_core = [statistics.mean(core) for core in zip(*usage_log)]
        print("\nAverage CPU usage per core (%):")
        for i, usage in enumerate(avg_usage_per_core):
            print(f"Core {i}: {usage:.2f}%")

        # Overall average CPU usage across all cores
        overall_avg = statistics.mean(avg_usage_per_core)
        print(f"\nOverall average CPU usage: {overall_avg:.2f}%")

    print("=====================================")

if __name__ == "__main__":
    # Example usage
    run_and_benchmark("Python scripts/filtbackproj.py", "Single-Core CT Reconstruction")
    run_and_benchmark("Python scripts/filtbackproj-multicore.py", "Multi-Core CT Reconstruction")
