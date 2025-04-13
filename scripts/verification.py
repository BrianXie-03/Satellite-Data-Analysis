import numpy as np
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
from netCDF4 import Dataset
import cartopy.feature as cfeature
from scipy.ndimage import map_coordinates
import scipy.stats as stats
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import xarray as xr

class Comparison:
    def __init__(self, ui):
        self.ui = ui  # Reference to the main UI

    def extract_bits(self, qc_data, start_bit, num_bits):
        qc_int = np.nan_to_num(qc_data, nan=0).astype(np.uint8)
        mask = ((1 << num_bits) - 1) << start_bit
        extracted_bits = (qc_int & mask) >> start_bit

        result = extracted_bits.astype(float)
        result[np.isnan(qc_data)] = np.nan
        return result
    
    #changed
    def compare_qc_flags(self, ref_qc, new_qc, output_dir):
        """详细比较BRF QC标志的每个位"""
        # BRF DQF标志位定义
        brf_dqf_bits = {
            'quality_score': {
                'start': 0,
                'bits': 3,
                'values': {
                    0: 'Good',
                    1: 'Snow',
                    2: 'Heavy aerosol (AOD>0.5)',
                    3: 'Fixed aerosol (AOD=0.05)',
                    4: 'Cloudy (not absolutely clear)',
                    5: 'Large SZA',
                    6: 'Large VZA',
                    7: 'Bad L1b'
                },
                'note': {
                    0: 'high quality',
                    1: 'high quality',
                    2: 'medium quality',
                    3: 'medium quality',
                    4: 'low quality',
                    5: 'invalid',
                    6: 'invalid',
                    7: 'invalid'
                }
            },
            'retrieval_path': {
                'start': 3,
                'bits': 2,
                'values': {
                    0: 'R1',
                    1: 'R2',
                    2: 'R3 (at least one band has no retrieval)',
                    3: 'R3 (at least one band has no retrieval)'
                },
                'note': 'R3 is the main subroutine for clear-sky, R1 is the backup subroutine'
            },
            'small_scattering_angle': {
                'start': 5,
                'bits': 1,
                'values': {
                    0: 'Scattering angle > 5 degrees',
                    1: 'Scattering angle < 5 degrees'
                },
                'note': 'Scattering angle to catch approximate hotspot scope'
            },
            'cloud': {
                'start': 6,
                'bits': 1,
                'values': {
                    0: 'Absolutely clear',
                    1: 'Probably clear, probably cloudy, absolutely cloudy'
                }
            },
            'aod_availability': {
                'start': 7,
                'bits': 1,
                'values': {
                    0: 'Valid AOD',
                    1: 'Invalid climatology'
                }
            }
        }

        
        # for flag_name, flag_info in brf_dqf_bits.items():
        #     self.plot_qc_comparison(ref_qc, new_qc, flag_name, flag_info, output_dir)
        
        results = {}
        total_valid_pixels = np.sum(~np.isnan(ref_qc))

        bit_start = int(self.ui.input_start_bit.text())
        bit_length = int(self.ui.input_bit_length.text())
        flag_name = ''

        if bit_start == 0:
            flag_name = 'quality_score'
        elif bit_start == 3:
            flag_name ='retrieval_path'
        elif bit_start == 5:
            flag_name = 'small_scattering_angle'
        elif bit_start == 6:
            flag_name = 'cloud'
        elif bit_start == 7:
            flag_name = 'aod_availability'

        flag_info = brf_dqf_bits[flag_name]
        self.plot_qc_comparison(ref_qc, new_qc, flag_name, flag_info, output_dir)

        ref_bits = self.extract_bits(ref_qc, bit_start, bit_length)
        new_bits = self.extract_bits(new_qc, bit_start, bit_length)
        
        valid_pixels = ~np.isnan(ref_bits)
        
        # 如果是aod_availability标记，进行采样分析
        # if flag_name == 'aod_availability':
            # total_diff = self.analyze_aod_availability_changes(ref_bits, new_bits, valid_pixels)
        
        value_stats = {}
        valid_pixels = ~np.isnan(ref_bits)
        
        matching_pixels = np.sum((ref_bits == new_bits) & valid_pixels)
        different_pixels = np.sum((ref_bits != new_bits) & valid_pixels)
        print(ref_bits)
        
        print(f"\n{flag_name.replace('_', ' ').title()}:")
        print(f"  Matching pixels: {matching_pixels}")
        print(f"  Different pixels: {different_pixels}")
        print(f"  Matching percentage: {(matching_pixels / total_valid_pixels * 100):.2f}%")
        
        print("  Value distribution:")
        for value in range(2**bit_length):

            ref_count = np.sum((ref_bits == value) & valid_pixels)
            new_count = np.sum((new_bits == value) & valid_pixels)
            matching = np.sum((ref_bits == value) & (new_bits == value) & valid_pixels)
            
            value_stats[value] = {
                'description': flag_info['values'][value],
                'ref_count': int(ref_count),
                'new_count': int(new_count),
                'matching': int(matching),
                'ref_percentage': (ref_count / total_valid_pixels * 100),
                'new_percentage': (new_count / total_valid_pixels * 100)
            }
            
            print(f"    Value {value} ({flag_info['values'][value]}):")
            print(f"      Reference: {ref_count} ({value_stats[value]['ref_percentage']:.2f}%)")
            print(f"      New: {new_count} ({value_stats[value]['new_percentage']:.2f}%)")
            print(f"      Matching: {matching}")
    
        results[flag_name] = {
            'matching_percentage': (matching_pixels / total_valid_pixels * 100),
            'total_pixels': total_valid_pixels,
            'matching_pixels': matching_pixels,
            'different_pixels': different_pixels,
            'value_stats': value_stats
        }


        return results

    def plot_qc_comparison(self, ref_qc, new_qc, flag_name, flag_info, output_dir):
        
        # Collecting bit information such as start and length
        bit_start = int(self.ui.input_start_bit.text())
        bit_length = int(self.ui.input_bit_length.text())
        
        # extract the specific bits for each of the files
        ref_bits = self.extract_bits(ref_qc, bit_start, bit_length)
        new_bits = self.extract_bits(new_qc, bit_start, bit_length)
        
        # find the difference between the bits
        diff = ref_bits - new_bits

        self.plot_qc_histogram(ref_bits, new_bits, diff, flag_name, flag_info, output_dir)
        
        fig = plt.figure(figsize=(6, 4))
        
        # Standard WGS84 
        semi_major = 6378137.0
        # Ellipsoid WGS84
        semi_minor = 6356752.31414
        # For GOES-East data
        longitude_of_projection_origin = -75.0
        # Normal for geostationary satellites (like GOES, Himawari, Meteosat)
        perspective_point_height = 3.5786023E7

        # Set up required parameter for projection type, in this case a globe with the major and minor axis specified above
        globe = ccrs.Globe(ellipse='sphere', semimajor_axis=semi_major, semiminor_axis=semi_minor)
        
        # established projection type (in this case its Geostationary with previously identified parameters)
        projection = ccrs.Geostationary(central_longitude=longitude_of_projection_origin, 
                                    satellite_height=perspective_point_height, globe=globe)
        

        # Identify the three data we are trying to gain
        titles = ['Reference', 'New', 'Difference']
        data_list = [ref_bits, new_bits, diff]
        
        # Establish the largest bit so if flag_info['bits'] = 3, then itll be 1000 - 1 = 0111 which is the bits used 
        max_value = (1 << flag_info['bits']) - 1
        cmaps = ['viridis', 'viridis', 'seismic']
        ranges = [(0, max_value), (0, max_value), (-max_value, max_value)]
        
        for idx, (title, data, cmap, value_range) in enumerate(zip(titles, data_list, cmaps, ranges)):
            fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': projection})

            img = ax.imshow(data, origin='upper', transform=projection,
                            extent=(-5434894.8823, 5434894.8823, -5434894.8823, 5434894.8823),
                            cmap=cmap, vmin=value_range[0], vmax=value_range[1])

            ax.gridlines(color='gray', alpha=0.5)
            ax.coastlines(resolution='50m', color='black', linestyle='--')

            cbar = plt.colorbar(img, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)

            if idx < 2:
                cbar.set_ticks(range(max_value + 1))
                # cbar.set_ticklabels([f"{i}\n({flag_info['values'][i]})" for i in range(max_value + 1)])

            ax.set_title(f'{title} - {flag_name.replace("_", " ").title()}', fontsize=14, pad=20)

            output_name = os.path.join(output_dir, f'BRF_{title}.png')
            plt.savefig(output_name, bbox_inches='tight')
            plt.close(fig)

    def plot_qc_histogram(self, ref_bits, new_bits, diff, flag_name, flag_info, output_dir):
        """Plot and save separate histograms for QC flag values."""

        # Flatten arrays and remove NaNs
        ref_values = ref_bits[~np.isnan(ref_bits)].flatten()
        new_values = new_bits[~np.isnan(new_bits)].flatten()
        diff_values = diff[~np.isnan(diff)].flatten()
        
        categories = list(flag_info['values'].keys())

        bins = np.arange(min(categories) - 0.5, max(categories) + 0.5, 1)

        # For the difference histogram, extend bin range to include negative values
        diff_min, diff_max = int(np.min(diff_values)), int(np.max(diff_values))
        diff_bins = np.arange(diff_min - 0.5, diff_max + 0.5, 1)

        datasets = [
            ('hist_file1', ref_values, 'blue', bins), 
            ('hist_file2', new_values, 'green', bins), 
            ('hist_difference', diff_values, 'red', diff_bins)
        ]

        for title, values, color, bin_range in datasets:
            fig, ax = plt.subplots(figsize=(5, 3))

            counts, bin_edges, _ = ax.hist(values, bins=bin_range, color=color, edgecolor='black', alpha=0.6, density=True)

            if title == "Difference":
                ax.set_xticks(range(diff_min, diff_max + 1)) 
            else:
                ax.set_xticks(categories)

            ax.set_title(f"{title} - {flag_name.replace('_', ' ').title()}")
            ax.set_xlabel("QC Value")
            ax.set_ylabel("Density")

            output_name = os.path.join(output_dir, f'{title}.png')
            plt.savefig(output_name, bbox_inches='tight')
            plt.close(fig) 

    def plot_histogram(self, file1, file2, output_path="results/brf_analysis"):
        d1 = xr.open_dataset(file1)
        d2 = xr.open_dataset(file2)

        channel1 = str(self.ui.fileDropdown1.currentText())
        channel2 = str(self.ui.fileDropdown2.currentText())

        data1 = d1[channel1][:].values.flatten()
        data2 = d2[channel2][:].values.flatten()

        if self.ui.qc_check.isChecked():
            bit_start = int(self.ui.input_start_bit.text())
            bit_length = int(self.ui.input_bit_length.text())

            qc1 = self.extract_bits(d1["Ref_QF"][:].values.flatten(), bit_start, bit_length)
            qc2 = self.extract_bits(d2["DQF"][:].values.flatten(), bit_start, bit_length)

            valid_mask = (qc1 == 0) & (qc2 == 0) & np.isfinite(data1) & np.isfinite(data2)

            data1 = data1[valid_mask]
            data2 = data2[valid_mask]
        else:
            valid_mask = np.isfinite(data1) & np.isfinite(data2)
            data1 = data1[valid_mask]
            data2 = data2[valid_mask]

        data_diff = data1 - data2

        # Create bins
        bins_common = np.linspace(min(data1.min(), data2.min()), max(data1.max(), data2.max()), 100)
        bins_diff = np.linspace(data_diff.min(), data_diff.max(), 100)

        # Save File 1 histogram
        plt.figure(figsize=(5, 4))
        plt.hist(data1, bins=bins_common, color='blue', alpha=0.7)
        plt.title("File 1 Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        file1_path = os.path.join(output_path, "hist_file1.png")
        plt.savefig(file1_path, bbox_inches='tight')
        plt.close()

        # Save File 2 histogram
        plt.figure(figsize=(5, 4))
        plt.hist(data2, bins=bins_common, color='orange', alpha=0.7)
        plt.title("File 2 Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        file2_path = os.path.join(output_path, "hist_file2.png")
        plt.savefig(file2_path, bbox_inches='tight')
        plt.close()

        # Save Difference histogram
        plt.figure(figsize=(5, 4))
        plt.hist(data_diff, bins=bins_diff, color='green', alpha=0.7)
        plt.title("Difference Histogram")
        plt.xlabel("Difference")
        plt.ylabel("Frequency")
        plt.grid(True)
        diff_path = os.path.join(output_path, "hist_difference.png")
        plt.savefig(diff_path, bbox_inches='tight')
        plt.close()

    def analyze_aod_availability_changes(self, ref_bits, new_bits, valid_mask):
        """分析AOD可用性标记的变化情况，并采样不同的像素"""
        print("\nAOD Availability Flag Change Analysis:")
        print("-----------------------------------")
        
        # 找到标记不同的像素
        diff_mask = (ref_bits != new_bits) & valid_mask
        diff_indices = np.where(diff_mask)
        total_diff = len(diff_indices[0])
        
        if total_diff > 0:
            # 计算采样间隔
            sample_size = min(10, total_diff)
            step = max(1, total_diff // sample_size)
            
            print(f"\nTotal pixels with different AOD availability: {total_diff}")
            print(f"Sampling {sample_size} pixels with step size {step}")
            print("\nSample pixels:")
            print("Format: [row, col] (1D index) - Reference -> New")
            print("        0: Valid AOD, 1: Invalid climatology")
            print("-" * 50)
            
            # 等间距采样
            for i in range(0, min(total_diff, sample_size * step), step):
                row = diff_indices[0][i]
                col = diff_indices[1][i]
                linear_index = row * ref_bits.shape[1] + col  # 计算一维索引
                
                ref_val = ref_bits[row, col]
                new_val = new_bits[row, col]
                
                print(f"[{row:4d}, {col:4d}] ({linear_index:7d}) - {ref_val} -> {new_val}")
        else:
            print("No differences found in AOD availability flag")
        
        return total_diff

    def compare_brf_files(self, file1, file2, output_dir, projection):
        nc1 = Dataset(file1, 'r')  # Reference file
        nc2 = Dataset(file2, 'r')  # New file

        file1_var = str(self.ui.fileDropdown1.currentText())
        file2_var = str(self.ui.fileDropdown2.currentText())
        ref_data = nc1[file1_var][:]
        new_data = nc2[file2_var][:]
        
        ref_mask = ((ref_data != nc1[file1_var]._FillValue) & (ref_data >= 0) & (ref_data <= 1))
        new_mask = ((new_data != nc2[file2_var]._FillValue) & (new_data >= 0) & (new_data <= 1))
        valid_mask = ref_mask & new_mask
        
        ref_valid = ref_data[valid_mask]
        new_valid = new_data[valid_mask]
        diff = np.where(valid_mask, ref_data - new_data, np.nan)
        
        results = {
            'ref_mean': np.mean(ref_valid),
            'new_mean': np.mean(new_valid),
            'mean_diff': np.nanmean(diff),
            'std_diff': np.nanstd(diff),
            'max_diff': np.nanmax(np.abs(diff)),
            'valid_pixels': np.sum(valid_mask),
            'relative_diff_percent': (np.nanmean(np.abs(diff)) / np.nanmean(np.abs(ref_valid))) * 100
        }
        
        self.plot_comparison(
            ref_data,
            new_data,
            diff,
            str(file1_var[2:3]),
            output_dir,
        )
        
        ref_qc = nc1['Ref_QF'][:] 
        new_qc = nc2['DQF'][:]
        
        ref_qc_masked = np.where(valid_mask, ref_qc, np.nan)
        new_qc_masked = np.where(valid_mask, new_qc, np.nan)

        if self.ui.qc_check.isChecked():
            qc_results = self.compare_qc_flags(ref_qc_masked, new_qc_masked, output_dir)
            all_results = {'reflectance': results, 'qc': qc_results}
        else:
            self.plot_histogram(file1, file2)
            all_results = {'reflectance': results}

        self.plot_scatter_plot(file1, file2)
        nc1.close()
        nc2.close()
        return all_results
        
    def plot_qc_line(self, value_stats, flag_name, output_path="qc_line_plot.png"):
        # Extract values and corresponding counts
        categories = list(value_stats.keys())  # QC value indices (e.g., 0, 1, 2, ...)
        descriptions = [value_stats[val]['description'] for val in categories]
        
        ref_counts = [value_stats[val]['ref_count'] for val in categories]
        new_counts = [value_stats[val]['new_count'] for val in categories]

        x_positions = np.arange(len(categories))

        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot reference and new counts
        ax.plot(x_positions, ref_counts, marker="o", linestyle="-", color="blue", label="Reference Count")
        ax.plot(x_positions, new_counts, marker="s", linestyle="-", color="red", label="New Count")

        # Fill space between the two lines
        ax.fill_between(x_positions, ref_counts, new_counts, color="gray", alpha=0.3)

        # Formatting
        ax.set_xticks(x_positions)
        ax.set_xticklabels(descriptions, rotation=30, ha="right")
        ax.set_ylabel("Pixel Count")
        ax.set_title(f"Reference vs New Pixel Count ({flag_name.replace('_', ' ').title()})")
        ax.legend()

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Line graph saved as {output_path}")

    def plot_scatter_plot(self, file1, file2, output_path = "scatterplot_comparison.png"):
        d1 = xr.open_dataset(file1)
        d2 = xr.open_dataset(file2)

        channel1 = str(self.ui.fileDropdown1.currentText())
        channel2 = str(self.ui.fileDropdown2.currentText())

        extract_data1 = d1[channel1][:].values.flatten()
        extract_data2 = d2[channel2][:].values.flatten()

        if self.ui.qc_check.isChecked():
            bit_start = int(self.ui.input_start_bit.text())
            bit_length = int(self.ui.input_bit_length.text())            

            d1_clean = self.extract_bits(d1["Ref_QF"][:].values.flatten(), bit_start, bit_length)
            d2_clean = self.extract_bits(d2["DQF"][:].values.flatten(), bit_start, bit_length)
            mask = (np.isfinite(d1_clean) & np.isfinite(d2_clean) )
            d1_clean = d1_clean[mask].astype(int)
            d2_clean = d2_clean[mask].astype(int)
        else:  
            mask = (~np.isnan(extract_data1)) & (~np.isnan(extract_data2)) 
            d1_clean = extract_data1[mask]
            d2_clean = extract_data2[mask]
            # & (extract_data1 >= 0) & (extract_data1 <= 20000) & (extract_data2 >= 0) & (extract_data2 <= 20000)

        print(d1_clean)
        print(d2_clean)
        plt.figure(figsize=(8, 5))
        plt.scatter(d1_clean, d2_clean, s=3, alpha=0.3, color='purple')
        plt.xlabel("file 1")
        plt.ylabel("file 2")
        plt.title("Scatter Plot Comparison: File 1 vs File 2")
        plt.grid(True)

        # 1:1 line for visual comparison
        min_val = min(d1_clean.min(), d2_clean.min())
        max_val = max(d1_clean.max(), d2_clean.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="1:1 line")
        # plt.legend()
        plt.axis('square')
        output_path = "/home/brian/research/results/brf_analysis/scatterplot_comparison.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Line graph saved as {output_path}")

    def plot_comparison(self, ref_data, new_data, diff_data, title, output_dir):
        """Plot reference, new, and difference data separately."""
        data_items = [
            ('Reference', ref_data, 'viridis', (np.nanmin(ref_data), np.nanmax(ref_data))),
            ('New', new_data, 'viridis', (np.nanmin(new_data), np.nanmax(new_data))),
            ('Difference', diff_data, 'seismic', (-np.nanmax(np.abs(diff_data)), np.nanmax(np.abs(diff_data))))
        ]

        # Projection setup
        semi_major = 6378137.0
        semi_minor = 6356752.31414
        longitude_of_projection_origin = -75.0
        perspective_point_height = 3.5786023E7
        globe = ccrs.Globe(ellipse='sphere', semimajor_axis=semi_major, semiminor_axis=semi_minor)
        projection = ccrs.Geostationary(central_longitude=longitude_of_projection_origin, 
                                        satellite_height=perspective_point_height, globe=globe)

        extent = (-5434894.8823, 5434894.8823, -5434894.8823, 5434894.8823)

        for label, data, cmap, vlim in data_items:
            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(1, 1, 1, projection=projection)

            img = ax.imshow(data, origin='upper', transform=projection, extent=extent,
                            cmap=cmap, vmin=vlim[0], vmax=vlim[1])
            
            ax.gridlines(color='gray', alpha=0.5)
            ax.coastlines(resolution='50m', color='black', linestyle='--')
            plt.colorbar(img, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
            ax.set_title(f'{label} - {title}', fontsize=14, pad=20)

            plt.tight_layout()
            output_name = os.path.join(output_dir, f'BRF_{label}.png')
            fig.savefig(output_name, bbox_inches='tight')
            plt.close()

    def save_results_to_file(results, output_dir):
        """将结果保存到文本文件"""
        with open(os.path.join(output_dir, 'brf_comparison_results.txt'), 'w') as f:
            # 保存反射率结果
            f.write("BRF Comparison Results\n")
            f.write("=====================\n\n")
            
            for band, stats in results['reflectance'].items():
                f.write(f"{band} Statistics:\n")
                f.write("-----------------\n")
                f.write(f"Reference mean: {stats['ref_mean']:.6f}\n")
                f.write(f"New mean: {stats['new_mean']:.6f}\n")
                f.write(f"Mean difference: {stats['mean_diff']:.6f}\n")
                f.write(f"Standard deviation: {stats['std_diff']:.6f}\n")
                f.write(f"Maximum absolute difference: {stats['max_diff']:.6f}\n")
                f.write(f"Valid pixels: {stats['valid_pixels']}\n")
                f.write(f"Relative difference: {stats['relative_diff_percent']:.2f}%\n\n")
            
            # 保存QC结果
            f.write("\nQC Flag Comparison Results\n")
            f.write("=========================\n\n")
            
            for flag_name, flag_results in results['qc'].items():
                f.write(f"\n{flag_name.replace('_', ' ').title()}:\n")
                f.write("-" * (len(flag_name) + 1) + "\n")
                f.write(f"Matching percentage: {flag_results['matching_percentage']:.2f}%\n")
                f.write(f"Total pixels: {flag_results['total_pixels']}\n")
                f.write(f"Different pixels: {flag_results['different_pixels']}\n\n")
                
                f.write("Value distribution:\n")
                for value, stats in flag_results['value_stats'].items():
                    f.write(f"\nValue {value} ({stats['description']}):\n")
                    f.write(f"  Reference count: {stats['ref_count']}\n")
                    f.write(f"  New count: {stats['new_count']}\n")
                    f.write(f"  Matching pixels: {stats['matching']}\n")
                    f.write(f"  Reference percentage: {stats['ref_percentage']:.2f}%\n")
                    f.write(f"  New percentage: {stats['new_percentage']:.2f}%\n")

    #changed
    # def plot_comparison(self, ref_data, new_data, diff_data, title, output_dir, projection_type):

    # #     def get_projection(projection_type):
    # #         semi_major = 6378137.0
    # #         semi_minor = 6356752.31414
    # #         longitude_of_projection_origin = -75.0
    # #         perspective_point_height = 3.5786023E7

    # #         if projection_type == "Geostationary":
    # #             globe = ccrs.Globe(ellipse='sphere', semimajor_axis=semi_major, semiminor_axis=semi_minor)
    # #             return ccrs.Geostationary(central_longitude=longitude_of_projection_origin, 
    # #                                     satellite_height=perspective_point_height, globe=globe)

    # #         elif projection_type == "PlateCarree: Simple latitude/longitude":
    # #             return ccrs.PlateCarree()

    # #         # elif projection_type == "Sinusoidal":
    # #         #     return ccrs.Sinusoidal(central_longitude=0)

    # #         # elif projection_type == "NorthPolarStereo":
    # #         #     return ccrs.NorthPolarStereo(central_longitude=0, true_scale_latitude=70)

    # #         # elif projection_type == "SouthPolarStereo":
    # #         #     return ccrs.SouthPolarStereo(central_longitude=0, true_scale_latitude=-71)

    # #         else:
    # #             raise ValueError(f"Unsupported projection type: {projection_type}")
        
    # #     projection = get_projection(projection_type)
    
    # #     titles = ['Reference', 'New', 'Difference']
    # #     data_list = [ref_data, new_data, diff_data]
    # #     cmaps = ['viridis', 'viridis', 'seismic']
        
    # #     # 计算数据范围
    # #     valid_min = min(np.nanmin(ref_data), np.nanmin(new_data))
    # #     valid_max = max(np.nanmax(ref_data), np.nanmax(new_data))
    # #     diff_range = max(abs(np.nanmin(diff_data)), abs(np.nanmax(diff_data)))
        
    # #     ranges = [(valid_min, valid_max), (valid_min, valid_max), (-diff_range, diff_range)]

    # #     fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': projection})
        
    # #        # Set extent based on projection type
    # #     if projection_type == "PlateCarree: Simple latitude/longitude":
    # #         extent = [-180, 180, -90, 90]
    # #     elif projection_type == "Geostationary":
    # #         extent = [-5434894.8823, 5434894.8823, -5434894.8823, 5434894.8823]
        
    # #     # Check for user-defined ROI
    # #     if self.ui.ROI_combo.currentText() == "Input max/min long/lat":
    # #         extent = [float(self.ui.Min_Long_Value.text()), 
    # #                 float(self.ui.Max_Long_Value.text()), 
    # #                 float(self.ui.Min_Lat_Value.text()), 
    # #                 float(self.ui.Max_lat_Value.text())]

    # #     # Create latitude/longitude grid (assuming data is on a regular lat/lon grid)
    # #     lons = np.linspace(extent[0], extent[1], ref_data.shape[1])
    # #     lats = np.linspace(extent[2], extent[3], ref_data.shape[0])
    # #     lon_grid, lat_grid = np.meshgrid(lons, lats)
    # #     file_names = ["ref", "new", "diff"]

    # #     for idx in range(3):
    # #         fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': projection})

    # #         # Use pcolormesh instead of imshow
    # #         img = ax.pcolormesh(lon_grid, lat_grid, data_list[idx], 
    # #                             transform=ccrs.PlateCarree(),
    # #                             cmap=cmaps[idx], vmin=ranges[idx][0], vmax=ranges[idx][1])

    # #         ax.set_extent(extent, crs=ccrs.PlateCarree())

    # #         # Add coastlines and gridlines
    # #         ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.8)
    # #         ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)

    # #         ax.set_title(f'{projection_type} | {titles[idx]}', fontsize=12, pad=10)
    # #         fig.colorbar(img, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)

    # #         # Save figure
    # #         output_name = os.path.join(output_dir, file_names[idx] + ".png")
    # #         fig.savefig(output_name, bbox_inches='tight', dpi=150)
    # #         plt.close(fig)
    #     return 1

    # def plot_qc_scatter(self, value_stats, flag_name, output_path="qc_proportion.png"):
#         categories = list(value_stats.keys())  # Extract QC values (0, 1, 2, ...)
#         descriptions = [value_stats[val]['description'] for val in categories]  # Descriptive labels

#         # Compute proportions (Reference in New)
#         proportions1 = [ (value_stats[val]['ref_percentage']) for val in categories]
#         proportions2 = [ (value_stats[val]['new_percentage']) for val in categories]

#         x_positions = np.arange(len(categories))  # Assign numerical positions

#         fig, ax = plt.subplots(figsize=(8, 5))

#         # Scatter plot
#         ax.scatter(x_positions, proportions1, color="green", marker="o", s=100, label="Proportion (%)")
#         ax.scatter(x_positions, proportions2, color="red", marker="o", s=100, label="Proportion (%)")

#         # Formatting
#         ax.set_xticks(x_positions)
#         ax.set_xticklabels(descriptions, rotation=25, ha="right")
#         ax.set_ylabel("Proportion of Reference in New (%)")
#         ax.set_title(f"QC Proportion Scatter Plot ({flag_name.replace('_', ' ').title()})")
#         ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5)  # Reference line at 100%
#         ax.legend()

#         # Save as PNG
#         plt.savefig(output_path, dpi=300, bbox_inches="tight")
#         plt.close(fig)
#         print(f"Scatter plot saved as {output_path}")