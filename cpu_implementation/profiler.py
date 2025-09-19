import cProfile
import pstats
import io
import pandas as pd
from PIL import Image
from filtbackproj_multicore_full_parallelization import *

def profile_to_dataframe(func, *args, **kwargs):
    """Profile a function and return the results as a DataFrame."""
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()
    
    # Capture stats to string
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats()
    
    # Parse the output (simplified)
    lines = s.getvalue().split('\n')
    data = []
    start_parsing = False
    for line in lines:
        if 'ncalls' in line and 'tottime' in line:
            start_parsing = True
            continue
        if start_parsing and line.strip() and not line.startswith(' '):
            break
        if start_parsing and line.strip():
            parts = line.split()
            if len(parts) >= 6:
                data.append({
                    'ncalls': parts[0],
                    'tottime': float(parts[1]) if parts[1].replace('.','').isdigit() else parts[1],
                    'percall': float(parts[2]) if parts[2].replace('.','').isdigit() else parts[2],
                    'cumtime': float(parts[3]) if parts[3].replace('.','').isdigit() else parts[3],
                    'percall2': float(parts[4]) if parts[4].replace('.','').isdigit() else parts[4],
                    'filename': ' '.join(parts[5:])
                })
    
    return pd.DataFrame(data), result


if __name__ == '__main__':
    # Number of cores for parallel processing inside functions
    n_jobs = 1
    
    # Load image
    myImgName = '004001_01_01_066'
    myImgPath = Image.open(f'data/phantoms/{myImgName}.png').convert('L')
    myImgPad, c0, c1 = padImage(myImgPath)
    theta = np.arange(0, 360.1, 0.1)
    
    # Profile each function with keyword arguments
    print("Profiling getProj...")
    getproj_df, mySino = profile_to_dataframe(
        getProj,
        img=myImgPad,
        theta=theta,
        n_jobs=n_jobs
    )
    
    print("Profiling projFilter...")
    filter_df, filtSino = profile_to_dataframe(
        projFilter,
        sino=mySino,
        n_jobs=n_jobs
    )
    
    print("Profiling backproject...")
    backproj_df, recon = profile_to_dataframe(
        backproject,
        sinogram=filtSino,
        theta=theta,
        n_jobs=n_jobs
    )
    
    # Save profiling results to Excel
    profiling_script = 'filtbackproj_multicore_full_parallelization'
    output_path = f'data/profiling_result/{myImgName}_{profiling_script}_{n_jobs}_3600.xlsx'
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        getproj_df.to_excel(writer, sheet_name='Forward_Projection', index=False)
        filter_df.to_excel(writer, sheet_name='Filtering', index=False)
        backproj_df.to_excel(writer, sheet_name='Backprojection', index=False)
    
    print(f"Results saved to {output_path}")
