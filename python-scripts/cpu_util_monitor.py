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
from filtbackproj_multicore import *
# from filtbackproj import *  # Import single-core version for comparison

class PerformanceMonitor:
    """Monitor CPU and memory usage during execution"""
    def __init__(self):
        self.cpu_data = []
        self.memory_data = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval=0.1):
        """Start monitoring system resources"""
        self.cpu_data = []
        self.memory_data = []
        self.monitoring = True
        
        def monitor():
            while self.monitoring:
                # CPU utilization per core
                cpu_percent = psutil.cpu_percent(interval=interval, percpu=True)
                self.cpu_data.append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_data.append({
                    'percent': memory.percent,
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3)
                })
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring and return collected data"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        # Calculate statistics
        cpu_array = np.array(self.cpu_data)
        cpu_stats = {
            'avg_total_utilization': np.mean(cpu_array),
            'avg_per_core': np.mean(cpu_array, axis=0),
            'max_per_core': np.max(cpu_array, axis=0),
            'cores_count': cpu_array.shape[1] if len(cpu_array) > 0 else 0
        }
        
        memory_df = pd.DataFrame(self.memory_data)
        memory_stats = {
            'peak_usage_percent': memory_df['percent'].max(),
            'peak_usage_gb': memory_df['used_gb'].max(),
            'avg_usage_percent': memory_df['percent'].mean()
        }
        
        return cpu_stats, memory_stats

def profile_with_monitoring(func, *args, **kwargs):
    """Profile function with system resource monitoring"""
    monitor = PerformanceMonitor()
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Profile execution
    pr = cProfile.Profile()
    pr.enable()
    start_time = time.time()
    
    result = func(*args, **kwargs)
    
    end_time = time.time()
    pr.disable()
    
    # Stop monitoring
    cpu_stats, memory_stats = monitor.stop_monitoring()
    
    # Process profiling data
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats()
    
    execution_time = end_time - start_time
    
    return {
        'result': result,
        'execution_time': execution_time,
        'cpu_stats': cpu_stats,
        'memory_stats': memory_stats,
        'profile_stats': s.getvalue()
    }

def calculate_efficiency_metrics(results, baseline_cores=1):
    """Calculate parallel efficiency metrics"""
    baseline_time = results[baseline_cores]['execution_time']
    
    metrics = {}
    for cores, data in results.items():
        speedup = baseline_time / data['execution_time']
        efficiency = speedup / cores if cores > 0 else 0
        
        metrics[cores] = {
            'execution_time': data['execution_time'],
            'speedup': speedup,
            'efficiency': efficiency,
            'cpu_utilization': data['cpu_stats']['avg_total_utilization'],
            'peak_memory_gb': data['memory_stats']['peak_usage_gb']
        }
    
    return metrics

def scalability_test():
    """Main scalability testing function"""
    # Test parameters
    core_counts = [1, 2, 4, 6, 8, -1]  # -1 means all available cores
    max_cores = psutil.cpu_count()
    
    # Replace -1 with actual core count
    core_counts = [max_cores if c == -1 else c for c in core_counts]
    core_counts = [c for c in core_counts if c <= max_cores]
    
    print(f"Testing scalability with core counts: {core_counts}")
    print(f"System has {max_cores} CPU cores available")
    
    # Load test image
    myImg = 'SheppLogan'
    print(f"Loading phantom: {myImg}")
    
    try:
        myImgPath = Image.open(f'data/phantoms/{myImg}.png').convert('L')
        myImgPad, c0, c1 = padImage(myImgPath)
        theta = np.arange(0, 361, 1)
        print(f"Image loaded: {myImgPad.size}, Angles: {len(theta)}")
        # myImgPath = Image.open(f'data/phantoms/{myImg}.png').convert('L')
        # myImgArray = np.array(myImgPath)  # Convert to numpy array
        # myImgPad, c0, c1 = padImage(myImgArray)
        # theta = np.arange(0, 361, 1)
        # print(f"Image loaded: {myImgPad.shape}, Angles: {len(theta)}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Generate projections once (same for all tests)
    print("Generating forward projections...")
    mySino = getProj(myImgPad, theta)
    
    # Test filtering scalability
    print("\n=== FILTERING SCALABILITY TEST ===")
    filter_results = {}
    
    for cores in core_counts:
        print(f"Testing filtering with {cores} cores...")
        
        if cores == 1:
            # Use single-core version from filtbackproj.py
            from filtbackproj import projFilter as projFilter_single
            result = profile_with_monitoring(projFilter_single, mySino)
        else:
            # Use multi-core version
            result = profile_with_monitoring(projFilter, mySino, n_jobs=cores)
        
        filter_results[cores] = result
        print(f"  Time: {result['execution_time']:.3f}s, "
              f"CPU: {result['cpu_stats']['avg_total_utilization']:.1f}%, "
              f"Memory: {result['memory_stats']['peak_usage_gb']:.2f}GB")
    
    # Test backprojection scalability
    print("\n=== BACKPROJECTION SCALABILITY TEST ===")
    backproj_results = {}
    
    # Use filtered sinogram from cores=4 test (or first available)
    test_cores = 4 if 4 in filter_results else list(filter_results.keys())[0]
    filtSino = filter_results[test_cores]['result']
    
    for cores in core_counts:
        print(f"Testing backprojection with {cores} cores...")
        
        if cores == 1:
            # Use single-core version
            from filtbackproj import backproject as backproject_single
            result = profile_with_monitoring(backproject_single, filtSino, theta)
        else:
            # Use multi-core version
            result = profile_with_monitoring(backproject, filtSino, theta, n_jobs=cores)
        
        backproj_results[cores] = result
        print(f"  Time: {result['execution_time']:.3f}s, "
              f"CPU: {result['cpu_stats']['avg_total_utilization']:.1f}%, "
              f"Memory: {result['memory_stats']['peak_usage_gb']:.2f}GB")
    
    # Calculate efficiency metrics
    print("\n=== CALCULATING EFFICIENCY METRICS ===")
    filter_metrics = calculate_efficiency_metrics(filter_results)
    backproj_metrics = calculate_efficiency_metrics(backproj_results)
    
    # Create comprehensive results dataframe
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
    
    # Save detailed results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    excel_filename = f'data/profiling_result/{myImg}_scalability_analysis_{timestamp}.xlsx'
    
    # with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    #     results_df.to_excel(writer, sheet_name='Summary', index=False)
        
    #     # Detailed filter metrics
    #     filter_detail_df = pd.DataFrame(filter_metrics).T
    #     filter_detail_df.to_excel(writer, sheet_name='Filter_Details')
        
    #     # Detailed backprojection metrics
    #     backproj_detail_df = pd.DataFrame(backproj_metrics).T
    #     backproj_detail_df.to_excel(writer, sheet_name='Backproj_Details')
    
    print(f"\nResults saved to: {excel_filename}")
    
    # Generate performance plots
    create_performance_plots(results_df, myImg, timestamp)
    
    # Print summary
    print_summary(results_df)
    
    return results_df

def create_performance_plots(results_df, image_name, timestamp):
    """Create performance visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Speedup curves
    axes[0,0].plot(results_df['cores'], results_df['filter_speedup'], 'o-', label='Filtering', linewidth=2, markersize=8)
    axes[0,0].plot(results_df['cores'], results_df['backproj_speedup'], 's-', label='Backprojection', linewidth=2, markersize=8)
    axes[0,0].plot(results_df['cores'], results_df['cores'], '--', alpha=0.5, label='Ideal Linear')
    axes[0,0].set_xlabel('Number of Cores')
    axes[0,0].set_ylabel('Speedup')
    axes[0,0].set_title('Speedup vs Number of Cores')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Efficiency curves
    axes[0,1].plot(results_df['cores'], results_df['filter_efficiency']*100, 'o-', label='Filtering', linewidth=2, markersize=8)
    axes[0,1].plot(results_df['cores'], results_df['backproj_efficiency']*100, 's-', label='Backprojection', linewidth=2, markersize=8)
    axes[0,1].axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='100% Efficiency')
    axes[0,1].set_xlabel('Number of Cores')
    axes[0,1].set_ylabel('Parallel Efficiency (%)')
    axes[0,1].set_title('Parallel Efficiency vs Number of Cores')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Execution time comparison
    axes[1,0].plot(results_df['cores'], results_df['filter_time'], 'o-', label='Filtering', linewidth=2, markersize=8)
    axes[1,0].plot(results_df['cores'], results_df['backproj_time'], 's-', label='Backprojection', linewidth=2, markersize=8)
    axes[1,0].plot(results_df['cores'], results_df['total_time'], '^-', label='Total', linewidth=2, markersize=8)
    axes[1,0].set_xlabel('Number of Cores')
    axes[1,0].set_ylabel('Execution Time (seconds)')
    axes[1,0].set_title('Execution Time vs Number of Cores')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_yscale('log')
    
    # Memory usage
    axes[1,1].plot(results_df['cores'], results_df['peak_memory_gb'], 'o-', color='red', linewidth=2, markersize=8)
    axes[1,1].set_xlabel('Number of Cores')
    axes[1,1].set_ylabel('Peak Memory Usage (GB)')
    axes[1,1].set_title('Memory Usage vs Number of Cores')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'data/profiling_result/{image_name}_scalability_plots_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Performance plots saved to: {plot_filename}")

def print_summary(results_df):
    """Print summary of results"""
    print("\n" + "="*60)
    print("SCALABILITY ANALYSIS SUMMARY")
    print("="*60)
    
    max_speedup_filter = results_df['filter_speedup'].max()
    max_speedup_backproj = results_df['backproj_speedup'].max()
    
    best_efficiency_filter = results_df.loc[results_df['filter_efficiency'].idxmax()]
    best_efficiency_backproj = results_df.loc[results_df['backproj_efficiency'].idxmax()]
    
    print(f"Maximum Speedup:")
    print(f"  Filtering: {max_speedup_filter:.2f}x")
    print(f"  Backprojection: {max_speedup_backproj:.2f}x")
    
    print(f"\nBest Efficiency:")
    print(f"  Filtering: {best_efficiency_filter['filter_efficiency']:.1%} at {best_efficiency_filter['cores']} cores")
    print(f"  Backprojection: {best_efficiency_backproj['backproj_efficiency']:.1%} at {best_efficiency_backproj['cores']} cores")
    
    print(f"\nExecution Time Range:")
    print(f"  Filtering: {results_df['filter_time'].min():.3f}s - {results_df['filter_time'].max():.3f}s")
    print(f"  Backprojection: {results_df['backproj_time'].min():.3f}s - {results_df['backproj_time'].max():.3f}s")
    
    print(f"\nMemory Usage Range: {results_df['peak_memory_gb'].min():.2f}GB - {results_df['peak_memory_gb'].max():.2f}GB")

if __name__ == '__main__':
    print("Starting CT Reconstruction Scalability Analysis...")
    print(f"System Information:")
    print(f"  CPU Cores: {psutil.cpu_count()}")
    print(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    print("-" * 60)
    
    try:
        results = scalability_test()
        print("\nScalability analysis completed successfully!")
    except Exception as e:
        print(f"Error during scalability testing: {e}")
        import traceback
        traceback.print_exc()