import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
from pathlib import Path

class HSIJPEGReader:
    def __init__(self, data_folder):
        """
        Initialize HSI JPEG reader
        
        Args:
            data_folder: Path to folder containing HSI slice images
        """
        self.data_folder = Path(data_folder)
        self.image_files = sorted(glob.glob(str(self.data_folder / "*.jpg")))
        self.hsi_cube = None
        
    def load_hsi_cube(self):
        """
        Load HSI data where height = spectral bands, width = spatial dimension
        Each JPEG file represents one spatial line scan
        
        Returns:
            numpy array: [spatial_lines, spatial_width, spectral_bands]
        """
        if not self.image_files:
            raise ValueError("No JPEG files found in the specified folder")
        
        # Read first image to get dimensions
        first_img = Image.open(self.image_files[0])
        img_array = np.array(first_img)
        
        if len(img_array.shape) == 3:  # RGB image
            first_img = first_img.convert('L')
            img_array = np.array(first_img)
            
        spectral_bands, spatial_width = img_array.shape  # height=spectral, width=spatial
        num_spatial_lines = len(self.image_files)
        
        print(f"Loading {num_spatial_lines} spatial line scans...")
        print(f"Spectral bands: {spectral_bands}")
        print(f"Spatial width: {spatial_width}")
        
        # Initialize HSI cube: [spatial_lines, spatial_width, spectral_bands]
        self.hsi_cube = np.zeros((num_spatial_lines, spatial_width, spectral_bands), dtype=np.uint8)
        
        # Load each spatial line
        for i, img_file in enumerate(self.image_files):
            img = Image.open(img_file)
            if img.mode != 'L':
                img = img.convert('L')
            
            img_array = np.array(img)
            # Transpose: height×width -> width×height (spatial×spectral)
            self.hsi_cube[i, :, :] = img_array.T
            
            if (i + 1) % 100 == 0:
                print(f"Loaded {i + 1}/{num_spatial_lines} spatial lines")
        
        print("HSI cube loaded successfully!")
        print(f"Final shape: [spatial_lines={num_spatial_lines}, spatial_width={spatial_width}, spectral_bands={spectral_bands}]")
        return self.hsi_cube
    
    def extract_pixel_spectrum(self, row, col):
        """
        Extract spectral signature for a specific pixel
        
        Args:
            row, col: Pixel coordinates
            
        Returns:
            numpy array: Spectral signature
        """
        if self.hsi_cube is None:
            raise ValueError("HSI cube not loaded. Call load_hsi_cube() first.")
            
        return self.hsi_cube[row, col, :]
    
    def extract_roi_spectrum(self, row_start, row_end, col_start, col_end):
        """
        Extract average spectrum from region of interest
        
        Args:
            row_start, row_end, col_start, col_end: ROI boundaries
            
        Returns:
            numpy array: Average spectral signature
        """
        if self.hsi_cube is None:
            raise ValueError("HSI cube not loaded. Call load_hsi_cube() first.")
            
        roi = self.hsi_cube[row_start:row_end, col_start:col_end, :]
        return np.mean(roi, axis=(0, 1))
    
    def apply_spectral_preprocessing(self, method='savgol'):
        """
        Apply spectral preprocessing to reduce noise
        
        Args:
            method: 'savgol', 'moving_average', 'gaussian'
        """
        from scipy.signal import savgol_filter
        from scipy.ndimage import gaussian_filter1d
        
        if self.hsi_cube is None:
            raise ValueError("HSI cube not loaded. Call load_hsi_cube() first.")
        
        print(f"Applying {method} preprocessing...")
        
        processed_cube = np.copy(self.hsi_cube).astype(np.float32)
        
        # Vectorized preprocessing - much faster
        if method == 'savgol':
            # Apply Savitzky-Golay along spectral axis
            processed_cube = savgol_filter(processed_cube, 5, 2, axis=2)
        
        elif method == 'moving_average':
            # Simple moving average using convolution
            from scipy.ndimage import uniform_filter1d
            processed_cube = uniform_filter1d(processed_cube, size=3, axis=2)
        
        elif method == 'gaussian':
            # Gaussian smoothing along spectral axis
            processed_cube = gaussian_filter1d(processed_cube, sigma=1.0, axis=2)
        
        self.hsi_cube = processed_cube
        print("Preprocessing completed!")
    
    def compute_spectral_derivatives(self):
        """
        Compute first and second derivatives of spectra
        
        Returns:
            tuple: (first_derivative, second_derivative) cubes
        """
        if self.hsi_cube is None:
            raise ValueError("HSI cube not loaded. Call load_hsi_cube() first.")
        
        # First derivative
        first_deriv = np.gradient(self.hsi_cube, axis=2)
        
        # Second derivative  
        second_deriv = np.gradient(first_deriv, axis=2)
        
        return first_deriv, second_deriv
    
    def visualize_spectra(self, pixel_coords=None, roi_coords=None, show_derivatives=False):
        """
        Visualize spectral signatures
        
        Args:
            pixel_coords: List of (row, col) tuples for individual pixels
            roi_coords: (row_start, row_end, col_start, col_end) for ROI
            show_derivatives: Whether to show derivative spectra
        """
        if self.hsi_cube is None:
            raise ValueError("HSI cube not loaded. Call load_hsi_cube() first.")
        
        fig, axes = plt.subplots(2 if show_derivatives else 1, 1, figsize=(12, 8))
        if not show_derivatives:
            axes = [axes]
        
        band_numbers = np.arange(self.hsi_cube.shape[2])
        
        # Plot original spectra
        if pixel_coords:
            for i, (row, col) in enumerate(pixel_coords):
                spectrum = self.extract_pixel_spectrum(row, col)
                axes[0].plot(band_numbers, spectrum, label=f'Pixel ({row},{col})')
        
        if roi_coords:
            spectrum = self.extract_roi_spectrum(*roi_coords)
            axes[0].plot(band_numbers, spectrum, label='ROI Average', linewidth=2)
        
        axes[0].set_xlabel('Spectral Band Number')
        axes[0].set_ylabel('Intensity (0-255)')
        axes[0].set_title('Spectral Signatures')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot derivatives if requested
        if show_derivatives:
            first_deriv, _ = self.compute_spectral_derivatives()
            
            if pixel_coords:
                for i, (row, col) in enumerate(pixel_coords):
                    deriv_spectrum = first_deriv[row, col, :]
                    axes[1].plot(band_numbers, deriv_spectrum, label=f'Pixel ({row},{col})')
            
            if roi_coords:
                roi_deriv = np.mean(first_deriv[roi_coords[0]:roi_coords[1], 
                                               roi_coords[2]:roi_coords[3], :], axis=(0,1))
                axes[1].plot(band_numbers, roi_deriv, label='ROI Average', linewidth=2)
            
            axes[1].set_xlabel('Spectral Band Number')
            axes[1].set_ylabel('First Derivative')
            axes[1].set_title('First Derivative Spectra')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_processed_data(self, output_file):
        """
        Save processed HSI cube to numpy format
        
        Args:
            output_file: Path for output .npy file
        """
        if self.hsi_cube is None:
            raise ValueError("HSI cube not loaded. Call load_hsi_cube() first.")
        
        np.save(output_file, self.hsi_cube)
        print(f"HSI cube saved to {output_file}")

# Example usage - analyze two plastic materials
def main():
    # Analyze first plastic material
    print("=== PLASTIC MATERIAL 1 ===")
    hsi_reader1 = HSIJPEGReader("data/Plastic HSI/20250905_2122")
    # hsi_reader1 = HSIJPEGReader("data/Plastic HSI/20250820_2126")
    hsi_cube1 = hsi_reader1.load_hsi_cube()
    
    # Analyze second plastic material  
    print("\n=== PLASTIC MATERIAL 2 ===")
    hsi_reader2 = HSIJPEGReader("data/Plastic HSI/20250905_2123")
    # hsi_reader2 = HSIJPEGReader("data/Plastic HSI/20250820_2126")
    hsi_cube2 = hsi_reader2.load_hsi_cube()
    
    # Extract average spectra from center regions (no preprocessing)
    h1, w1 = hsi_cube1.shape[:2]
    h2, w2 = hsi_cube2.shape[:2]
    
    # Use center 50x50 pixel regions
    plastic1_spectrum = hsi_reader1.extract_roi_spectrum(h1//2-25, h1//2+25, w1//2-25, w1//2+25)
    plastic2_spectrum = hsi_reader2.extract_roi_spectrum(h2//2-25, h2//2+25, w2//2-25, w2//2+25)
    
    # Compare raw spectra
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(plastic1_spectrum, label='Plastic 1', linewidth=2)
    plt.plot(plastic2_spectrum, label='Plastic 2', linewidth=2)
    plt.xlabel('Spectral Band')
    plt.ylabel('Intensity (0-255)')
    plt.title('Raw Spectral Comparison')
    plt.legend()
    plt.grid(True)
    
    # Show difference
    plt.subplot(1, 2, 2)
    diff = plastic1_spectrum - plastic2_spectrum
    plt.plot(diff, 'r-', linewidth=2)
    plt.xlabel('Spectral Band') 
    plt.ylabel('Intensity Difference')
    plt.title('Spectral Difference (Material 1 - Material 2)')
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Print basic stats
    print(f"\nMaterial 1 cube shape: {hsi_cube1.shape}")
    print(f"Material 2 cube shape: {hsi_cube2.shape}")
    print(f"Spectral range: 0-{max(plastic1_spectrum.max(), plastic2_spectrum.max())}")
    print(f"Max difference: {abs(diff).max():.1f}")
    
    return hsi_reader1, hsi_reader2

if __name__ == "__main__":
    main()