import os
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def rename_raw_files(directory_path):
    """
    Rename *.raw files to zero-padded 4-digit format (e.g., 1.raw -> 0001.raw)
    and save them to a 'renamed/' subdirectory
    
    Args:
        directory_path: Path to the directory containing .raw files
    """
    # Get all .raw files in the directory
    raw_files = glob.glob(os.path.join(directory_path, "*.raw"))
    raw_files.extend(glob.glob(os.path.join(directory_path, "*.RAW")))
    
    if not raw_files:
        logger.warning(f"No .raw files found in {directory_path}")
        return
    
    # Sort files numerically by extracting the number from filename
    def get_file_number(filepath):
        try:
            base_name = os.path.basename(filepath)
            file_name = os.path.splitext(base_name)[0]
            return int(file_name)
        except ValueError:
            return float('inf')  # Non-numeric files go to the end
    
    raw_files.sort(key=get_file_number)
    
    logger.info(f"Found {len(raw_files)} .raw files")
    logger.info(f"File range: {os.path.basename(raw_files[0])} to {os.path.basename(raw_files[-1])}")
    
    # Create 'renamed/' subdirectory
    output_dir = os.path.join(directory_path, "renamed")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    renamed_count = 0
    
    for file_path in raw_files:
        # Get the file name without extension
        base_name = os.path.basename(file_path)
        file_name = os.path.splitext(base_name)[0]
        extension = os.path.splitext(base_name)[1]
        
        # Check if the filename is numeric
        try:
            file_number = int(file_name)
        except ValueError:
            logger.warning(f"Skipping {base_name} - filename is not a number")
            continue
        
        # Check if number exceeds 4 digits
        if file_number > 9999:
            logger.warning(f"Skipping {base_name} - number exceeds 4-digit limit (9999)")
            continue
        
        # Create new filename with zero-padding
        new_file_name = f"{file_number:04d}{extension}"
        new_file_path = os.path.join(output_dir, new_file_name)
        
        # Check if target file already exists
        if os.path.exists(new_file_path):
            logger.warning(f"Skipping {base_name} - target file {new_file_name} already exists")
            continue
        
        # Copy the file with new name
        try:
            import shutil
            shutil.copy2(file_path, new_file_path)
            logger.info(f"Copied: {base_name} -> renamed/{new_file_name}")
            renamed_count += 1
        except Exception as e:
            logger.error(f"Failed to copy {base_name}: {e}")
    
    logger.info(f"Renaming complete. {renamed_count} files copied to 'renamed/' directory.")

if __name__ == "__main__":
    # Set your directory path here
    directory_path = "data/20251119_Tako_Wire_and_SiC/slices/"
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        logger.error(f"Directory not found: {directory_path}")
        logger.info("Please update the 'directory_path' variable in the script")
    else:
        rename_raw_files(directory_path)