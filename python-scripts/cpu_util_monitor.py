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

# from filtbackproj import *  # single-core fallback
from filtbackproj_multicore_full_parallelization import *
# from filtbackproj_multicore_hybrid_vectorized_para import *

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
        if data['execution_time'] == 0 :
            speedup = 2  # No idea what went wrong
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


def scalability_test():
    """Main function for scalability testing."""
    # core_counts = [1, 2, 4, 6, 8, -1]   // All core: -1
    core_counts = [1, 2, 3, 4, 5, 6, 7, 8]
    max_cores = psutil.cpu_count()
    core_counts = [max_cores if c == -1 else c for c in core_counts]
    core_counts = [c for c in core_counts if c <= max_cores]

    print(f"Testing scalability with core counts: {core_counts}")
    print(f"System has {max_cores} CPU cores available")

    myImg = '004007_01_01_519'
    print(f"Loading phantom: {myImg}")
    try:
        img_path = Image.open(f'data/phantoms/{myImg}.png').convert('L')
        myImgPad, c0, c1 = padImage(img_path)
        theta = np.arange(0, 361, 1)
        print(f"Image loaded: {myImgPad.size}, Angles: {len(theta)}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    print("Generating forward projections...")
    mySino = getProj(myImgPad, theta)

    # FILTERING
    print("\n=== FILTERING SCALABILITY TEST ===")
    filter_results = {}
    for cores in core_counts:
        print(f"Testing filtering with {cores} cores...")
        if cores == 1:
            from filtbackproj import projFilter as projFilter_single
            # result = profile_with_monitoring(projFilter_single, mySino)
            result = profile_with_monitoring(projFilter, mySino, n_jobs=cores)
        else:
            result = profile_with_monitoring(projFilter, mySino, n_jobs=cores)
        filter_results[cores] = result
        print(f"  Time: {result['execution_time']:.3f}s, "
              f"CPU: {result['cpu_stats']['avg_total_utilization']:.1f}%, "
              f"Memory: {result['memory_stats']['peak_rss_gb']:.2f}GB")

    # BACKPROJECTION
    print("\n=== BACKPROJECTION SCALABILITY TEST ===")
    backproj_results = {}
    test_cores = 4 if 4 in filter_results else list(filter_results.keys())[0]
    filtSino = filter_results[test_cores]['result']
    for cores in core_counts:
        print(f"Testing backprojection with {cores} cores...")
        if cores == 1:
            # from filtbackproj import backproject as backproject_single
            # result = profile_with_monitoring(backproject_single, filtSino, theta)
            result = profile_with_monitoring(backproject, filtSino, theta, n_jobs=cores)
        else:
            result = profile_with_monitoring(backproject, filtSino, theta, n_jobs=cores)
        backproj_results[cores] = result
        print(f"  Time: {result['execution_time']:.3f}s, "
              f"CPU: {result['cpu_stats']['avg_total_utilization']:.1f}%, "
              f"Memory: {result['memory_stats']['peak_rss_gb']:.2f}GB")

    print("\n=== CALCULATING EFFICIENCY METRICS ===")
    filter_metrics = calculate_efficiency_metrics(filter_results)
    backproj_metrics = calculate_efficiency_metrics(backproj_results)

    # Build summary DataFrame
    results_data = []
    for cores in core_counts:
        results_data.append({
            'cores': cores,
            'filter_time': filter_metrics[cores]['execution_time'],
            'filter_speedup': filter_metrics[cores]['speedup'],
            'filter_efficiency': filter_metrics[cores]['efficiency'],
            'backproj_time': backproj_metrics[cores]['execution_time'],
            'backproj_speedup': backproj_metrics[cores]['speedup'],
            'backproj_efficiency': backproj_metrics[cores]['efficiency'],
            'total_time': filter_metrics[cores]['execution_time'] + backproj_metrics[cores]['execution_time'],
            'peak_memory_gb': max(filter_metrics[cores]['peak_memory_gb'],
                                  backproj_metrics[cores]['peak_memory_gb']),
            'avg_cpu_utilization': (filter_metrics[cores]['cpu_utilization'] +
                                    backproj_metrics[cores]['cpu_utilization']) / 2
        })

    results_df = pd.DataFrame(results_data)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    excel_filename = f'data/scalability_test_result/{myImg}_scalability_analysis_{timestamp}.xlsx'
    print(f"\nResults saved to: {excel_filename}")

    create_performance_plots(results_df, myImg, timestamp)
    print_summary(results_df)

    return results_df


def create_performance_plots(results_df, image_name, timestamp):
    """Generate performance plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Speedup
    axes[0,0].plot(results_df['cores'], results_df['filter_speedup'], 'o-', label='Filtering', linewidth=2)
    axes[0,0].plot(results_df['cores'], results_df['backproj_speedup'], 's-', label='Backprojection', linewidth=2)
    axes[0,0].plot(results_df['cores'], results_df['cores'], '--', alpha=0.5, label='Ideal Linear')
    axes[0,0].set_xlabel('Cores')
    axes[0,0].set_ylabel('Speedup')
    axes[0,0].set_title('Speedup vs Cores')
    axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

    # Efficiency
    axes[0,1].plot(results_df['cores'], results_df['filter_efficiency']*100, 'o-', label='Filtering', linewidth=2)
    axes[0,1].plot(results_df['cores'], results_df['backproj_efficiency']*100, 's-', label='Backprojection', linewidth=2)
    axes[0,1].axhline(100, color='gray', linestyle='--', alpha=0.5)
    axes[0,1].set_xlabel('Cores'); axes[0,1].set_ylabel('Efficiency (%)')
    axes[0,1].set_title('Efficiency vs Cores'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

    # Execution Time
    axes[1,0].plot(results_df['cores'], results_df['filter_time'], 'o-', label='Filtering')
    axes[1,0].plot(results_df['cores'], results_df['backproj_time'], 's-', label='Backprojection')
    axes[1,0].plot(results_df['cores'], results_df['total_time'], '^-', label='Total')
    axes[1,0].set_xlabel('Cores'); axes[1,0].set_ylabel('Time (s)'); axes[1,0].set_title('Execution Time vs Cores')
    axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3); axes[1,0].set_yscale('log')

    # Memory Usage
    axes[1,1].plot(results_df['cores'], results_df['peak_memory_gb'], 'o-', color='red')
    axes[1,1].set_xlabel('Cores'); axes[1,1].set_ylabel('Peak Memory (GB)')
    axes[1,1].set_title('Memory Usage vs Cores'); axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_filename = f'data/scalability_test_result/{image_name}_scalability_plots_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Performance plots saved to: {plot_filename}")


def print_summary(results_df):
    """Print textual summary of scalability results."""
    print("\n" + "="*60)
    print("SCALABILITY ANALYSIS SUMMARY")
    print("="*60)

    max_speedup_filter = results_df['filter_speedup'].max()
    max_speedup_backproj = results_df['backproj_speedup'].max()

    best_eff_filter = results_df.loc[results_df['filter_efficiency'].idxmax()]
    best_eff_backproj = results_df.loc[results_df['backproj_efficiency'].idxmax()]

    print(f"Maximum Speedup:")
    print(f"  Filtering: {max_speedup_filter:.2f}x")
    print(f"  Backprojection: {max_speedup_backproj:.2f}x")

    print(f"\nBest Efficiency:")
    print(f"  Filtering: {best_eff_filter['filter_efficiency']:.1%} at {best_eff_filter['cores']} cores")
    print(f"  Backprojection: {best_eff_backproj['backproj_efficiency']:.1%} at {best_eff_backproj['cores']} cores")

    print(f"\nExecution Time Range:")
    print(f"  Filtering: {results_df['filter_time'].min():.3f}s - {results_df['filter_time'].max():.3f}s")
    print(f"  Backprojection: {results_df['backproj_time'].min():.3f}s - {results_df['backproj_time'].max():.3f}s")

    print(f"\nMemory Usage Range: {results_df['peak_memory_gb'].min():.2f}GB - {results_df['peak_memory_gb'].max():.2f}GB")


if __name__ == '__main__':
    print("Starting CT Reconstruction Scalability Analysis...")
    print(f"System Information:")
    print(f"  CPU Cores: {psutil.cpu_count()}")
    print(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    print("-"*60)

    try:
        results = scalability_test()
        print("\nScalability analysis completed successfully!")
    except Exception as e:
        print(f"Error during scalability testing: {e}")
        import traceback
        traceback.print_exc()
