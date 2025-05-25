import dask
import numpy as np
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
import xarray as xr
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import dask.array as da


class Comparison:
    def __init__(self, ui):
        self.ui = ui  # Reference to the main UI
        self.cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='8GB')
        self.client = Client(self.cluster)
        
    def extract_bits(self, qc_data, start_bit, num_bits):
        qc_int = np.nan_to_num(qc_data, nan=0).astype(np.uint8)
        mask = ((1 << num_bits) - 1) << start_bit
        extracted_bits = (qc_int & mask) >> start_bit

        result = extracted_bits.astype(float)
        result[np.isnan(qc_data)] = np.nan
        return result 

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

    # def plot_qc_histogram(self, ref_bits, new_bits, diff, flag_name, flag_info, output_dir):
    #     """Plot and save separate histograms for QC flag values."""

    #     # Flatten arrays and remove NaNs
    #     ref_values = ref_bits.data.compute().values
    #     new_values = new_bits.data.compute().values
    #     diff_values = diff.data.compute().values

    #     ref_values = ref_values[np.isfinite(ref_values)].flatten()
    #     new_values = new_values[np.isfinite(new_values)].flatten()
    #     diff_values = diff_values[np.isfinite(diff_values)].flatten()
        
    #     categories = list(flag_info['values'].keys())

    #     bins = np.arange(min(categories) - 0.5, max(categories) + 0.5, 1)

    #     # For the difference histogram, extend bin range to include negative values
    #     diff_min, diff_max = int(np.min(diff_values)), int(np.max(diff_values))
    #     diff_bins = np.arange(diff_min - 0.5, diff_max + 0.5, 1)

    #     datasets = [
    #         ('hist_file1', ref_values, 'blue', bins), 
    #         ('hist_file2', new_values, 'green', bins), 
    #         ('hist_difference', diff_values, 'red', diff_bins)
    #     ]

    #     for title, values, color, bin_range in datasets:
    #         fig, ax = plt.subplots(figsize=(5, 3))

    #         counts, bin_edges, _ = ax.hist(values, bins=bin_range, color=color, edgecolor='black', alpha=0.6, density=True)

    #         if title == "Difference":
    #             ax.set_xticks(range(diff_min, diff_max + 1)) 
    #         else:
    #             ax.set_xticks(categories)

    #         ax.set_title(f"{title} - {flag_name.replace('_', ' ').title()}")
    #         ax.set_xlabel("QC Value")
    #         ax.set_ylabel("Density")

    #         output_name = os.path.join(output_dir, f'{title}.png')
    #         plt.savefig(output_name, bbox_inches='tight')
    #         plt.close(fig) 


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


    # -------------------------------------------------------------------------------------------------------#
    # QC Histogram for Sinusoidal Code 
    #
    # def plot_qc_histogram(self, ref_bits, new_bits, diff, flag_name, flag_info, output_dir):
    
    #     # Use only a chunk of the data (e.g., first 1000 values) for faster processing
    #     ref_values = ref_bits.data[:1000].compute().values  # Take a slice (first 1000 values)
    #     new_values = new_bits.data[:1000].compute().values  # Take a slice (first 1000 values)
    #     diff_values = diff.data[:1000].compute().values    # Take a slice (first 1000 values)

    #     # Flatten and remove NaNs
    #     ref_values = ref_values[np.isfinite(ref_values)].flatten()
    #     new_values = new_values[np.isfinite(new_values)].flatten()
    #     diff_values = diff_values[np.isfinite(diff_values)].flatten()
        
    #     categories = list(flag_info['values'].keys())

    #     bins = np.arange(min(categories) - 0.5, max(categories) + 0.5, 1)

    #     # For the difference histogram, extend bin range to include negative values
    #     diff_min, diff_max = int(np.min(diff_values)), int(np.max(diff_values))
    #     diff_bins = np.arange(diff_min - 0.5, diff_max + 0.5, 1)

    #     datasets = [
    #         ('hist_file1', ref_values, 'blue', bins), 
    #         ('hist_file2', new_values, 'green', bins), 
    #         ('hist_difference', diff_values, 'red', diff_bins)
    #     ]

    #     for title, values, color, bin_range in datasets:
    #         fig, ax = plt.subplots(figsize=(5, 3))

    #         counts, bin_edges, _ = ax.hist(values, bins=bin_range, color=color, edgecolor='black', alpha=0.6, density=True)

    #         if title == "Difference":
    #             ax.set_xticks(range(diff_min, diff_max + 1)) 
    #         else:
    #             ax.set_xticks(categories)

    #         ax.set_title(f"{title} - {flag_name.replace('_', ' ').title()}")
    #         ax.set_xlabel("QC Value")
    #         ax.set_ylabel("Density")

    #         # Save the plot to the specified output directory
    #         os.makedirs(output_dir, exist_ok=True)
    #         output_name = os.path.join(output_dir, f'{title}.png')
    #         plt.savefig(output_name, bbox_inches='tight')
    #         plt.close(fig)

    # -------------------------------------------------------------------------------------------------------#

    # def plot_histogram(self, file1, file2, output_path="results/brf_analysis"):
    #     d1 = xr.open_dataset(file1, chunks={})
    #     d2 = xr.open_dataset(file2, chunks={})

    #     channel1 = str(self.ui.fileDropdown1.currentText())
    #     channel2 = str(self.ui.fileDropdown2.currentText())

    #     data1 = d1[channel1][:].compute().flatten().values
    #     data2 = d2[channel2][:].compute().flatten().values


    #     if self.ui.qc_check.isChecked():
    #         bit_start = int(self.ui.input_start_bit.text())
    #         bit_length = int(self.ui.input_bit_length.text())

    #         qc1 = self.extract_bits(d1["Ref_QF"][:].flatten().compute().values, bit_start, bit_length)
    #         qc2 = self.extract_bits(d2["DQF"][:].flatten().compute().values, bit_start, bit_length)

    #         valid_mask = (qc1 == 0) & (qc2 == 0) & np.isfinite(data1) & np.isfinite(data2)

    #         data1 = data1[valid_mask]
    #         data2 = data2[valid_mask]
    #     else:
    #         valid_mask = np.isfinite(data1) & np.isfinite(data2)
    #         data1 = data1[valid_mask]
    #         data2 = data2[valid_mask]

    #     data_diff = data1 - data2

    #     # Create bins
    #     bins_common = np.linspace(min(data1.min(), data2.min()), max(data1.max(), data2.max()), 100)
    #     bins_diff = np.linspace(data_diff.min(), data_diff.max(), 100)

    #     # Save File 1 histogram
    #     plt.figure(figsize=(5, 4))
    #     plt.hist(data1, bins=bins_common, color='blue', alpha=0.7)
    #     plt.title("File 1 Histogram")
    #     plt.xlabel("Value")
    #     plt.ylabel("Frequency")
    #     plt.grid(True)
    #     file1_path = os.path.join(output_path, "hist_file1.png")
    #     plt.savefig(file1_path, bbox_inches='tight')
    #     plt.close()

    #     # Save File 2 histogram
    #     plt.figure(figsize=(5, 4))
    #     plt.hist(data2, bins=bins_common, color='orange', alpha=0.7)
    #     plt.title("File 2 Histogram")
    #     plt.xlabel("Value")
    #     plt.ylabel("Frequency")
    #     plt.grid(True)
    #     file2_path = os.path.join(output_path, "hist_file2.png")
    #     plt.savefig(file2_path, bbox_inches='tight')
    #     plt.close()

    #     # Save Difference histogram
    #     plt.figure(figsize=(5, 4))
    #     plt.hist(data_diff, bins=bins_diff, color='green', alpha=0.7)
    #     plt.title("Difference Histogram")
    #     plt.xlabel("Difference")
    #     plt.ylabel("Frequency")
    #     plt.grid(True)
    #     diff_path = os.path.join(output_path, "hist_difference.png")
    #     plt.savefig(diff_path, bbox_inches='tight')
    #     plt.close()

    # def plot_histogram(self, file1, file2, output_path="results/brf_analysis"):
    #     # Open the datasets with chunks
    #     # d1 = xr.open_dataset(file1, chunks={})
    #     # d2 = xr.open_dataset(file2, chunks={})

    #     # # Select channels
    #     # channel1 = str(self.ui.fileDropdown1.currentText())
    #     # channel2 = str(self.ui.fileDropdown2.currentText())

    #     # # Get data from the selected channels, using a subset of the data (e.g., first 1000 elements)
    #     # data1 = d1[channel1].isel({list(d1[channel1].dims)[0]: slice(0, 5000)}).compute().values
    #     # data2 = d2[channel2].isel({list(d2[channel2].dims)[0]: slice(0, 5000)}).compute().values
    #     # data1 = data1.ravel()
    #     # data2 = data2.ravel()
    #     # print(data1)
    #     # print(data2)

    #     d1 = xr.open_dataset(file1, chunks={})
    #     d2 = xr.open_dataset(file2, chunks={})

    #     # Select channels
    #     channel1 = str(self.ui.fileDropdown1.currentText())
    #     channel2 = str(self.ui.fileDropdown2.currentText())

    #     # Get data from the selected channels, using a subset of the data (e.g., first 5000 elements)
    #     data1 = d1[channel1].isel({list(d1[channel1].dims)[0]: slice(2000, 4000)}).compute().values
    #     data2 = d2[channel2].isel({list(d2[channel2].dims)[0]: slice(2000, 4000)}).compute().values
    #     data1 = data1.ravel()
    #     data2 = data2.ravel()
    #     print(data1)
    #     print(data2)
    #     # Sample a smaller chunk for unique calculation (limit to 10,000 values max)
    #     max_sample = 1000

    #     if data1.size > max_sample:
    #         sample_indices1 = np.random.choice(data1.size, size=max_sample, replace=False)
    #         sampled_data1 = data1[sample_indices1]
    #     else:
    #         sampled_data1 = data1

    #     if data2.size > max_sample:
    #         sample_indices2 = np.random.choice(data2.size, size=max_sample, replace=False)
    #         sampled_data2 = data2[sample_indices2]
    #     else:
    #         sampled_data2 = data2

    #     # Get the first 500 unique values from each sampled dataset
    #     filtered_data1 = sampled_data1[np.isfinite(sampled_data1)]
    #     filtered_data2 = sampled_data2[np.isfinite(sampled_data2)]


    #     unique_sample1 = np.unique(filtered_data1)
    #     unique_sample2 = np.unique(filtered_data2)

    #     min_len = min(len(unique_sample1), len(unique_sample2))
    #     unique_sample1 = unique_sample1[:min_len]
    #     unique_sample2 = unique_sample2[:min_len]

    #     print(unique_sample1)
    #     print(unique_sample2)

    #     data1 = unique_sample1[:500]
    #     data2 = unique_sample2[:500]


    #     # # Use up to 500 unique values only
    #     # data1 = unique_data1[:500]
    #     # data2 = unique_data2[:500]

    #     # QC check: extract bits and apply valid mask
    #     if self.ui.qc_check.isChecked():
    #         bit_start = int(self.ui.input_start_bit.text())
    #         bit_length = int(self.ui.input_bit_length.text())

    #         qc1 = self.extract_bits(d1["Ref_QF"][:1000].compute().flatten().values, bit_start, bit_length)  # QC on first 1000 values
    #         qc2 = self.extract_bits(d2["DQF"][:1000].compute().flatten().values, bit_start, bit_length)  # QC on first 1000 values

    #         valid_mask = (qc1 == 0) & (qc2 == 0) & np.isfinite(data1) & np.isfinite(data2)

    #         data1 = data1[valid_mask]
    #         data2 = data2[valid_mask]
    #     else:
    #         valid_mask = np.isfinite(data1) & np.isfinite(data2)
    #         data1 = data1[valid_mask]
    #         data2 = data2[valid_mask]

    #     # Calculate difference between the two datasets
    #     data_diff = data2 - data1

    #     # Create bins for the histograms
    #     bins_common = np.linspace(min(data1.min(), data2.min()), max(data1.max(), data2.max()), 100)
    #     bins_diff = np.linspace(data_diff.min(), data_diff.max(), 100)

    #     # Save File 1 histogram
    #     plt.figure(figsize=(5, 4))
    #     plt.hist(data1, bins=bins_common, color='blue', alpha=0.7)
    #     plt.title("File 1 Histogram")
    #     plt.xlabel("Value")
    #     plt.ylabel("Frequency")
    #     plt.grid(True)
    #     file1_path = os.path.join(output_path, "hist_file1.png")
    #     plt.savefig(file1_path, bbox_inches='tight')
    #     plt.close()

    #     # Save File 2 histogram
    #     plt.figure(figsize=(5, 4))
    #     plt.hist(data2, bins=bins_common, color='orange', alpha=0.7)
    #     plt.title("File 2 Histogram")
    #     plt.xlabel("Value")
    #     plt.ylabel("Frequency")
    #     plt.grid(True)
    #     file2_path = os.path.join(output_path, "hist_file2.png")
    #     plt.savefig(file2_path, bbox_inches='tight')
    #     plt.close()

    #     # Save Difference histogram
    #     plt.figure(figsize=(5, 4))
    #     plt.hist(data_diff, bins=bins_diff, color='green', alpha=0.7)
    #     plt.title("Difference Histogram")
    #     plt.xlabel("Difference")
    #     plt.ylabel("Frequency")
    #     plt.grid(True)
    #     diff_path = os.path.join(output_path, "hist_difference.png")
    #     plt.savefig(diff_path, bbox_inches='tight')
    #     plt.close()

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

    # def compare_brf_files(self, file1, file2, output_dir, projection):
    #     # Open the datasets with dask for lazy loading
    #     nc1 = xr.open_dataset(file1, chunks='auto')
    #     nc2 = xr.open_dataset(file2, chunks='auto')

    #     file1_var = str(self.ui.fileDropdown1.currentText())
    #     file2_var = str(self.ui.fileDropdown2.currentText())        
    #     # Use dask arrays for lazy computation
    #     # ref_data = nc1[file1_var][:].values
    #     # new_data = nc2[file2_var][:].values

    #     ref_data = nc1[file1_var][:]
    #     new_data = nc2[file2_var][:]

    #     # Apply masks to ensure valid data (NaNs are masked)
    #     # ref_mask = np.isfinite(ref_data)
    #     # new_mask = np.isfinite(new_data)
    #     # valid_mask = ref_mask & new_mask
    #     valid_mask = xr.ufuncs.isfinite(ref_data) & xr.ufuncs.isfinite(new_data)

    #     # ref_valid = da.where(valid_mask, ref_data, da.nan)
    #     # new_valid = da.where(valid_mask, new_data, da.nan)
    #     ref_valid = ref_data.where(valid_mask)
    #     new_valid = new_data.where(valid_mask)

    #     # diff = ref_valid - new_valid
    #     diff = np.where(valid_mask, ref_data - new_data, da.nan)

    #     # Perform the computations
    #     results = {
    #         'ref_mean': ref_valid.mean().compute(),
    #         'new_mean': new_valid.mean().compute(),
    #         'mean_diff': diff.mean().compute(),
    #         'std_diff': diff.std().compute(),
    #         'max_diff': diff.max().compute(),
    #         'valid_pixels': valid_mask.sum().compute(),
    #         'relative_diff_percent': da.maximum(0, (diff.mean() / ref_valid.mean()) * 100).compute()
    #     }
    #     print("here 5")

    #     # Call the plot_comparison function
    #     self.plot_comparison(
    #         ref_data,
    #         new_data,
    #         diff,
    #         str(file1_var[2:3]),
    #         output_dir,
    #     )
        
    #     if self.ui.qc_check.isChecked():
    #         qc_results = self.compare_qc_flags(ref_valid, new_valid, output_dir)
    #         all_results = {'reflectance': results, 'qc': qc_results}
    #     else:
    #         self.plot_histogram(file1, file2)
    #         all_results = {'reflectance': results}

    #     self.plot_scatter_plot(file1, file2)
    #     print("Done!")
    #     nc1.close()
    #     nc2.close()
    #     return all_results

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

    # def plot_scatter_plot(self, file1, file2, output_path = "scatterplot_comparison.png"):
    #     d1 = xr.open_dataset(file1, chunks={'Rows': 1000, 'Columns': 2000})
    #     d2 = xr.open_dataset(file2, chunks={'Rows': 1000, 'Columns': 2000})

    #     channel1 = str(self.ui.fileDropdown1.currentText())
    #     channel2 = str(self.ui.fileDropdown2.currentText())

    #     var1 = d1[channel1]
    #     var2 = d2[channel2]

    #     # Initialize result lists
    #     d1_clean = []
    #     d2_clean = []

    #     N = 10000  # Max number of valid points to collect
    #     collected = 0

    #     # Iterate over blocks of data using .chunks
    #     for i in range(var1.sizes['Rows']):
    #         for j in range(var1.sizes['Columns']):
    #             if collected >= N:
    #                 break
    #             # Pull 1x1 slice (or small tile if needed)
    #             chunk1 = var1.isel(Rows=i, Columns=j).compute().values
    #             chunk2 = var2.isel(Rows=i, Columns=j).compute().values

    #             if np.isfinite(chunk1) and np.isfinite(chunk2):
    #                 d1_clean.append(chunk1.item())
    #                 d2_clean.append(chunk2.item())
    #                 collected += 1
    #         if collected >= N:
    #             break

    #     d1_clean = np.array(d1_clean)
    #     d2_clean = np.array(d2_clean)

    #     # Optional QC bit extraction
    #     if self.ui.qc_check.isChecked():
    #         bit_start = int(self.ui.input_start_bit.text())
    #         bit_length = int(self.ui.input_bit_length.text())

    #         qc1 = self.extract_bits(d1[channel1].values.ravel(), bit_start, bit_length)
    #         qc2 = self.extract_bits(d2[channel2].values.ravel(), bit_start, bit_length)

    #         valid_mask = (qc1 == 0) & (qc2 == 0)
    #         d1_clean = d1_clean[valid_mask[:len(d1_clean)]]
    #         d2_clean = d2_clean[valid_mask[:len(d2_clean)]]

    #     # Plotting
    #     plt.figure(figsize=(8, 5))
    #     plt.scatter(d1_clean, d2_clean, s=3, alpha=0.3, color='purple')
    #     plt.xlabel("File 1")
    #     plt.ylabel("File 2")
    #     plt.title("Scatter Plot Comparison")
    #     plt.grid(True)

    #     if len(d1_clean) > 0 and len(d2_clean) > 0:
    #         min_val = min(d1_clean.min(), d2_clean.min())
    #         max_val = max(d1_clean.max(), d2_clean.max())
    #         plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="1:1 line")
    #         plt.axis('square')

    #     # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #     plt.savefig(output_path, bbox_inches='tight')
    #     print(f"Scatter plot saved as {output_path}")
    #     plt.close()

# test code 

        # channel1 = str(self.ui.fileDropdown1.currentText())
        # channel2 = str(self.ui.fileDropdown2.currentText())
        # # Instead of loading all data into memory, process the data in chunks
        # data1 = d1[channel1]
        # data2 = d2[channel2]
        
        # # Initialize lists to store cleaned data
        # d1_clean = []
        # d2_clean = []

        # # Iterate over chunks of data
        # for i in range(len(data1['time'])):  # Assuming 'time' dimension is available
        #     chunk1 = data1.isel(time=i).values.flatten()
        #     chunk2 = data2.isel(time=i).values.flatten()

        #     # Mask and clean the data within the chunk
        #     mask = np.isfinite(chunk1) & np.isfinite(chunk2)
        #     d1_clean.extend(chunk1[mask])
        #     d2_clean.extend(chunk2[mask])

        # # Convert lists to numpy arrays for plotting
        # extract_data1 = np.array(d1_clean)
        # extract_data2 = np.array(d2_clean)
        
        # # extract_data1 = d1[channel1][:].values.flatten()
        # # extract_data2 = d2[channel2][:].values.flatten()

        # if self.ui.qc_check.isChecked():
        #     bit_start = int(self.ui.input_start_bit.text())
        #     bit_length = int(self.ui.input_bit_length.text())            

        #     d1_clean = self.extract_bits(d1[channel1][:].values.flatten(), bit_start, bit_length)
        #     d2_clean = self.extract_bits(d2[channel2][:].values.flatten(), bit_start, bit_length)
        #     mask = (np.isfinite(d1_clean) & np.isfinite(d2_clean) )
        #     d1_clean = d1_clean[mask].astype(int)
        #     d2_clean = d2_clean[mask].astype(int)
        # else:  
        #     mask = (~np.isnan(extract_data1)) & (~np.isnan(extract_data2)) 
        #     d1_clean = extract_data1[mask]
        #     d2_clean = extract_data2[mask]

        # print(d1_clean)
        # print(d2_clean)
        # plt.figure(figsize=(8, 5))
        # plt.scatter(d1_clean, d2_clean, s=3, alpha=0.3, color='purple')
        # plt.xlabel("file 1")
        # plt.ylabel("file 2")
        # plt.title("Scatter Plot Comparison: File 1 vs File 2")
        # plt.grid(True)

        # # 1:1 line for visual comparison
        # min_val = min(d1_clean.min(), d2_clean.min())
        # max_val = max(d1_clean.max(), d2_clean.max())
        # plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="1:1 line")
        # plt.axis('square')
        # output_path = "/home/brian/research/results/brf_analysis/scatterplot_comparison.png"
        # plt.savefig(output_path, bbox_inches='tight')
        # print(f"Line graph saved as {output_path}")

    def plot_comparison(self, ref_data, new_data, diff_data, title, output_dir):

        print("In plot comparison")

        choice = self.ui.comboBox.currentIndex()

        if choice == 0:
            data_items = [
            ('Reference', ref_data, 'viridis', (np.nanmin(ref_data), np.nanmax(ref_data))),
            ('New', new_data, 'viridis', (np.nanmin(new_data), np.nanmax(new_data))),
            ('Difference', diff_data, 'seismic', (-np.nanmax(np.abs(diff_data)), np.nanmax(np.abs(diff_data))))
            ]

            for label, data, cmap, vlim in data_items:
                fig = plt.figure(figsize=(8, 4))
                ax = fig.add_subplot(1, 1, 1)

                # Plot the raw data
                img = ax.imshow(data, origin='upper')

                # title
                ax.set_title(f'{label} - {title}', fontsize=14, pad=20)

                # Save figure
                os.makedirs(output_dir, exist_ok=True)
                output_name = os.path.join(output_dir, f'BRF_{label}.png')
                plt.tight_layout()
                fig.savefig(output_name, bbox_inches='tight')
                plt.close()

        elif choice == 1:
            #platecarree
            data_items = [
                ('Reference', ref_data, 'viridis', (np.nanmin(ref_data), np.nanmax(ref_data))),
                ('New', new_data, 'viridis', (np.nanmin(new_data), np.nanmax(new_data))),
                ('Difference', diff_data, 'seismic', (-np.nanmax(np.abs(diff_data)), np.nanmax(np.abs(diff_data))))
                ]

                # Use PlateCarree projection for regular lat/lon data
            projection = ccrs.PlateCarree()
            transform = ccrs.PlateCarree()

            # Full global extent in lat/lon
            extent = (-180, 180, -90, 90)

            for label, data, cmap, vlim in data_items:
                fig = plt.figure(figsize=(8, 4))
                ax = fig.add_subplot(1, 1, 1, projection=projection)

                ax.set_global()  # Display entire globe
                ax.set_extent(extent, crs=transform)

                # Plot data
                img = ax.imshow(data, origin='upper', extent=extent, transform=transform,
                                cmap=cmap, vmin=vlim[0], vmax=vlim[1])

                # Add map features
                ax.coastlines(resolution='110m', color='black', linestyle='-')
                ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
                ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')

                # Colorbar and title
                plt.colorbar(img, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
                ax.set_title(f'{label} - {title}', fontsize=14, pad=20)

                # Save figure
                os.makedirs(output_dir, exist_ok=True)
                output_name = os.path.join(output_dir, f'BRF_{label}.png')
                plt.tight_layout()
                fig.savefig(output_name, bbox_inches='tight')
                plt.close()
        elif choice == 2:
            #Sinusoidal
            ref_data_downsampled = ref_data.coarsen(Rows=10, Columns=10, boundary='trim').mean()
            new_data_downsampled = new_data.coarsen(Rows=10, Columns=10, boundary='trim').mean()
            diff_data_downsampled = diff_data.coarsen(Rows=10, Columns=10, boundary='trim').mean


            projection = ccrs.Sinusoidal()
            extent = (-180, 180, -90, 90)
            data_items = [
                ('Reference', ref_data_downsampled, 'viridis', (np.nanmin(ref_data_downsampled), np.nanmax(ref_data_downsampled))),
                ('New', new_data_downsampled, 'viridis', (np.nanmin(new_data_downsampled), np.nanmax(new_data_downsampled))),
                ('Difference', diff_data_downsampled, 'seismic', (-np.nanmax(np.abs(diff_data_downsampled)), np.nanmax(np.abs(diff_data_downsampled))))
            ]
            for label, data, cmap, vlim in data_items:
                fig = plt.figure(figsize=(8, 4))
                ax = fig.add_subplot(1, 1, 1, projection=projection)

                ax.set_extent(extent, crs=projection)

                # Plot data
                img = ax.imshow(data, origin='upper', interpolation='nearest', extent=extent, transform=projection,
                                cmap=cmap, vmin=vlim[0], vmax=vlim[1])

                # Add map features
                ax.coastlines(resolution='110m', color='black', linestyle='-')
                ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
                ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')

                # Colorbar and title
                plt.colorbar(img, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
                ax.set_title(f'{label} - {title}', fontsize=14, pad=20)

                # Save figure
                os.makedirs(output_dir, exist_ok=True)
                output_name = os.path.join(output_dir, f'BRF_{label}.png')
                plt.tight_layout()
                fig.savefig(output_name, bbox_inches='tight')
                plt.close()
            # pass
        elif choice == 3:
            # Polar Stereographic
            pass
        elif choice == 4:
            # Geostationary
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
        print("Done plot comparison")

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