import pandas as pd
import cProfile
import pstats
import io
from filtbackproj import *

def profile_to_dataframe(func, *args, **kwargs):
    """Profile function and return results as DataFrame"""
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
    
    # Find the data section
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

# Usage
if __name__ == '__main__':
    # Load image
    myImg = Image.open('data/phantoms/SheppLogan.png').convert('L')
    myImgPad, c0, c1 = padImage(myImg)
    theta = np.arange(0, 361, 1)
    
    # Profile each function
    print("Profiling getProj...")
    getproj_df, mySino = profile_to_dataframe(getProj, myImgPad, theta)
    
    print("Profiling projFilter...")
    filter_df, filtSino = profile_to_dataframe(projFilter, mySino)
    
    print("Profiling backproject...")
    backproj_df, recon = profile_to_dataframe(backproject, filtSino, theta)
    
    # Create Excel file with multiple sheets
    profiling_script = 'filtbackproj'
    with pd.ExcelWriter(f'data/profiling_result/{profiling_script}.xlsx', engine='openpyxl') as writer:
        getproj_df.to_excel(writer, sheet_name='Forward_Projection', index=False)
        filter_df.to_excel(writer, sheet_name='Filtering', index=False)
        backproj_df.to_excel(writer, sheet_name='Backprojection', index=False)
    
    print(f"Results saved to data/profiling_result/{profiling_script}.xlsx")