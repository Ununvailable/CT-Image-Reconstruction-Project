import cProfile
import pstats
import io
import pandas as pd
import numpy as np
import time
import psutil
import threading
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import filtbackproj_multicore_full_parallelization as tb_para
import filtbackproj_multicore_hybrid_vectorized_para as pb_para


class PerformanceMonitor:
    """Monitor CPU and per-process memory usage."""
    def __init__(self):
        self.cpu_data = []
        self.memory_data = []
        self.monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process()

    def start_monitoring(self, interval=0.1):
        """Start monitoring CPU and per-process memory usage."""
        self.cpu_data.clear()
        self.memory_data.clear()
        self.monitoring = True

        def monitor():
            while self.monitoring:
                cpu_percent = psutil.cpu_percent(interval=interval, percpu=True)
                self.cpu_data.append(cpu_percent)
                mem_info = self.process.memory_info()
                self.memory_data.append({
                    'rss_gb': mem_info.rss / (1024 ** 3),
                    'vms_gb': mem_info.vms / (1024 ** 3)
                })
                time.sleep(interval)

        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop monitoring and calculate stats."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

        cpu_array = np.array(self.cpu_data)
        cpu_stats = {
            'avg_total_utilization': np.mean(cpu_array),
            'avg_per_core': np.mean(cpu_array, axis=0),
            'max_per_core': np.max(cpu_array, axis=0),
            'cores_count': cpu_array.shape[1] if len(cpu_array) > 0 else 0
        }

        memory_df = pd.DataFrame(self.memory_data)
        memory_stats = {
            'peak_rss_gb': memory_df['rss_gb'].max(),
            'peak_vms_gb': memory_df['vms_gb'].max(),
            'avg_rss_gb': memory_df['rss_gb'].mean()
        }

        return cpu_stats, memory_stats


def profile_with_monitoring(func, *args, **kwargs):
    """Profile a function while monitoring CPU and per-process memory."""
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    pr = cProfile.Profile()
    pr.enable()
    start_time = time.time()

    result = func(*args, **kwargs)

    end_time = time.time()
    pr.disable()
    cpu_stats, memory_stats = monitor.stop_monitoring()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats()

    return {
        'result': result,
        'execution_time': end_time - start_time,
        'cpu_stats': cpu_stats,
        'memory_stats': memory_stats,
        'profile_stats': s.getvalue()
    }


def calculate_efficiency_metrics(results, baseline_cores=1):
    """Calculate speedup and parallel efficiency."""
    baseline_time = results[baseline_cores]['execution_time']
    metrics = {}
    for cores, data in results.items():
        if data['execution_time'] == 0:
            speedup = 2
        else:
            speedup = baseline_time / data['execution_time']
        efficiency = speedup / cores if cores > 0 else 0
        metrics[cores] = {
            'execution_time': data['execution_time'],
            'speedup': speedup,
            'efficiency': efficiency,
            'cpu_utilization': data['cpu_stats']['avg_total_utilization'],
            'peak_memory_gb': data['memory_stats']['peak_rss_gb']
        }
    return metrics


def scalability_test(backproj_module, implementation_name):
    """Test backprojection scalability for a given implementation."""
    core_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # core_counts = [1, 2, 3, 4, 5, 6, 7, 8]
    max_cores = psutil.cpu_count()
    core_counts = [c for c in core_counts if c <= max_cores]

    print(f"\n{'='*60}")
    print(f"Testing {implementation_name}")
    print(f"{'='*60}")
    print(f"Core counts: {core_counts}")

    myImg = '004007_01_01_519'
    print(f"Loading phantom: {myImg}")
    try:
        img_path = Image.open(f'data/phantoms/{myImg}.png').convert('L')
        myImgPad, c0, c1 = backproj_module.padImage(img_path)
        theta = np.arange(0, 361, 1)
        print(f"Image loaded: {myImgPad.size}, Angles: {len(theta)}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    print("Generating forward projections...")
    mySino = backproj_module.getProj(myImgPad, theta)

    print("Filtering projections...")
    filtSino = backproj_module.projFilter(mySino, n_jobs=max_cores)

    print(f"\n=== BACKPROJECTION SCALABILITY TEST ({implementation_name}) ===")
    backproj_results = {}
    for cores in core_counts:
        print(f"Testing backprojection with {cores} cores...", end=" ")
        result = profile_with_monitoring(backproj_module.backproject, filtSino, theta, n_jobs=cores)
        backproj_results[cores] = result
        print(f"Time: {result['execution_time']:.3f}s, "
              f"CPU: {result['cpu_stats']['avg_total_utilization']:.1f}%, "
              f"Memory: {result['memory_stats']['peak_rss_gb']:.2f}GB")

    metrics = calculate_efficiency_metrics(backproj_results)
    return metrics, core_counts


def create_comparison_plots(metrics_tb, metrics_pb, cores_tb, cores_pb, timestamp):
    """Generate comparison plots for thread-based and process-based implementations."""
<<<<<<< HEAD
    # fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    title_size = 18
    label_size = 14
=======
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=100)
    title_size = 20
    label_size = 18
    tick_size = 11
>>>>>>> 90fb1e34d55e1a821a316c31a92dbe91df5a0fe8

    # Extract data
    cores_t = sorted(metrics_tb.keys())
    speedup_t = [metrics_tb[c]['speedup'] for c in cores_t]
    usage_t = [metrics_tb[c]['cpu_utilization'] for c in cores_t]
    efficiency_t = [metrics_tb[c]['efficiency'] * 100 for c in cores_t]
    memory_t = [metrics_tb[c]['peak_memory_gb'] for c in cores_t]

    cores_p = sorted(metrics_pb.keys())
    speedup_p = [metrics_pb[c]['speedup'] for c in cores_p]
    usage_p = [metrics_pb[c]['cpu_utilization'] for c in cores_p]
    efficiency_p = [metrics_pb[c]['efficiency'] * 100 for c in cores_p]
    memory_p = [metrics_pb[c]['peak_memory_gb'] for c in cores_p]

    # Speedup vs Cores
    axes[0].plot(cores_t, speedup_t, 'o-', label='Thread-Based', linewidth=2.5, markersize=7, color='#1f77b4')
    axes[0].plot(cores_p, speedup_p, 's-', label='Process-Based', linewidth=2.5, markersize=7, color='#ff7f0e')
    axes[0].plot(cores_t, cores_t, '--', alpha=0.4, label='Ideal Linear', color='gray', linewidth=2)
    axes[0].set_xlabel('Cores', fontsize=label_size, fontweight='bold')
    axes[0].set_ylabel('Speedup', fontsize=label_size, fontweight='bold')
    axes[0].set_title('Speedup vs Cores', fontsize=title_size, fontweight='bold')
    axes[0].legend(fontsize=10, loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(labelsize=tick_size)
    axes[0].set_xlim(0, max(cores_t) + 0.5)
    axes[0].set_ylim(0, max(cores_t) + 1)

    # Individual Core Usage
    axes[1].plot(cores_t, usage_t, 'o-', label='Thread-Based', linewidth=2.5, markersize=7, color='#1f77b4')
    axes[1].plot(cores_p, usage_p, 's-', label='Process-Based', linewidth=2.5, markersize=7, color='#ff7f0e')
    axes[1].axhline(100, color='gray', linestyle='--', alpha=0.4, linewidth=2, label='Ideal')
    axes[1].set_xlabel('Cores', fontsize=label_size, fontweight='bold')
    axes[1].set_ylabel('Usage (%)', fontsize=label_size, fontweight='bold')
    axes[1].set_title('Individual Core Usage', fontsize=title_size, fontweight='bold')
    axes[1].legend(fontsize=10, loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(labelsize=tick_size)
    axes[1].set_xlim(0, max(cores_t) + 0.5)
    axes[1].set_ylim(0, 110)

    # Memory Usage vs Cores
    axes[2].plot(cores_t, memory_t, 'o-', label='Thread-Based', linewidth=2.5, markersize=7, color='#d62728')
    axes[2].plot(cores_p, memory_p, 's-', label='Process-Based', linewidth=2.5, markersize=7, color='#ff9800')
    axes[2].set_xlabel('Cores', fontsize=label_size, fontweight='bold')
    axes[2].set_ylabel('Peak Memory (GB)', fontsize=label_size, fontweight='bold')
    axes[2].set_title('Memory Usage vs Cores', fontsize=title_size, fontweight='bold')
    axes[2].legend(fontsize=10, loc='best')
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(labelsize=tick_size)
    axes[2].set_xlim(0, max(cores_t) + 0.5)

    # Consistent layout
    fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.12, wspace=0.28)
    
    fig.savefig(f'comparison_{timestamp}.png', dpi=100, bbox_inches='tight')
    plt.show()

    # plot_filename = f'data/scalability_test_result/backprojection_comparison_{timestamp}.png'
    # plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    # print(f"\nComparison plots saved to: {plot_filename}")


def print_comparison_summary(metrics_tb, metrics_pb):
    """Print comparison summary of thread-based and process-based implementations."""
    print("\n" + "="*70)
    print("BACKPROJECTION SCALABILITY COMPARISON SUMMARY")
    print("="*70)

    max_speedup_tb = max(m['speedup'] for m in metrics_tb.values())
    max_speedup_pb = max(m['speedup'] for m in metrics_pb.values())

    best_eff_tb = max(metrics_tb.values(), key=lambda x: x['efficiency'])
    best_eff_pb = max(metrics_pb.values(), key=lambda x: x['efficiency'])

    print(f"\nMaximum Speedup:")
    print(f"  Thread-Based: {max_speedup_tb:.2f}x")
    print(f"  Process-Based: {max_speedup_pb:.2f}x")

    print(f"\nBest Efficiency:")
    print(f"  Thread-Based: {best_eff_tb['efficiency']:.1%}")
    print(f"  Process-Based: {best_eff_pb['efficiency']:.1%}")

    exec_times_tb = [m['execution_time'] for m in metrics_tb.values()]
    exec_times_pb = [m['execution_time'] for m in metrics_pb.values()]

    print(f"\nExecution Time Range:")
    print(f"  Thread-Based: {min(exec_times_tb):.3f}s - {max(exec_times_tb):.3f}s")
    print(f"  Process-Based: {min(exec_times_pb):.3f}s - {max(exec_times_pb):.3f}s")

    mem_tb = [m['peak_memory_gb'] for m in metrics_tb.values()]
    mem_pb = [m['peak_memory_gb'] for m in metrics_pb.values()]

    print(f"\nMemory Usage Range:")
    print(f"  Thread-Based: {min(mem_tb):.2f}GB - {max(mem_tb):.2f}GB")
    print(f"  Process-Based: {min(mem_pb):.2f}GB - {max(mem_pb):.2f}GB")


if __name__ == '__main__':
    print("="*70)
    print("CT RECONSTRUCTION BACKPROJECTION SCALABILITY COMPARISON")
    print("="*70)
    print(f"System Information:")
    print(f"  CPU Cores: {psutil.cpu_count()}")
    print(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    print("-"*70)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    try:
        metrics_tb, cores_tb = scalability_test(tb_para, "Thread-Based Full Parallelization")
        metrics_pb, cores_pb = scalability_test(pb_para, "Process-Based Hybrid Vectorized")

        if metrics_tb and metrics_pb:
            print_comparison_summary(metrics_tb, metrics_pb)
            create_comparison_plots(metrics_tb, metrics_pb, cores_tb, cores_pb, timestamp)
            print("\nComparison analysis completed successfully!")
        else:
            print("Error: One or both implementations failed.")

    except Exception as e:
        print(f"Error during scalability testing: {e}")
        import traceback
        traceback.print_exc()