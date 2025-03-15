from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scripts.verification import Comparison

def extract_bits(self, qc_data, start_bit, num_bits):
    qc_int = np.nan_to_num(qc_data, nan=255).astype(np.uint8)
    mask = ((1 << num_bits) - 1) << start_bit
    extracted_bits = (qc_int & mask) >> start_bit

    result = extracted_bits.astype(float)
    result[np.isnan(qc_data)] = np.nan
    return result


def get_projection(nc_file):
    with Dataset(nc_file, 'r') as nc:
        for attr in nc.ncattrs():
            if "projection" in attr.lower() or "spatial" in attr.lower():
                print(f"{nc_file} Projection: {getattr(nc, attr)}")

# Example usage
get_projection("data_files/G16_ABI_FD_2025011_1900_00_LAND_SFC_REFLECTANCE_EN.nc")
get_projection("data_files/DR_ABI-L2-BRFF-M6_G16_s20250111900205_e20250111909513_c20250111914544.nc")

nc1 = Dataset("data_files/G16_ABI_FD_2025011_1900_00_LAND_SFC_REFLECTANCE_EN.nc", 'r')  # Reference file
nc2 = Dataset("data_files/DR_ABI-L2-BRFF-M6_G16_s20250111900205_e20250111909513_c20250111914544.nc", 'r')  # New file
file1_var = "Ch1_Ref"
file2_var = "BRF1"
ref_data = nc1[file1_var][:]
new_data = nc2[file2_var][:]
ref_mask = ((ref_data != nc1[file1_var]._FillValue) & 
            (ref_data >= 0) & 
            (ref_data <= 1))
new_mask = ((new_data != nc2[file2_var]._FillValue) & 
            (new_data >= 0) & 
            (new_data <= 1))
valid_mask = ref_mask & new_mask

ref_valid = (ref_data[valid_mask])
new_valid = (new_data[valid_mask])
diff = np.nanmean(np.where(valid_mask, ref_data - new_data, np.nan))
ref_valid = np.mean(ref_data[valid_mask])
new_valid = np.mean(new_data[valid_mask])

# data = ('qc', {'quality_score': {'matching_percentage': np.float64(82.00511921481596), 'total_pixels': np.int64(2747687), 'matching_pixels': np.int64(2253244), 'different_pixels': np.int64(494443), 'value_stats': {0: {'description': 'Good', 'ref_count': 622493, 'new_count': 961497, 'matching': 622469, 'ref_percentage': np.float64(22.655164143514163), 'new_percentage': np.float64(34.99295953287256)}, 1: {'description': 'Snow', 'ref_count': 0, 'new_count': 0, 'matching': 0, 'ref_percentage': np.float64(0.0), 'new_percentage': np.float64(0.0)}, 2: {'description': 'Heavy aerosol (AOD>0.5)', 'ref_count': 22936, 'new_count': 178330, 'matching': 22936, 'ref_percentage': np.float64(0.8347384545619643), 'new_percentage': np.float64(6.490186109262081)}}}})
# print(data[1]['quality_score']['value_stats'][0])
# print(data[1]['quality_score']['value_stats'][1])
# print(data[1]['quality_score']['value_stats'][2])
# data_dict = data[1]['quality_score']['value_stats']/[0]

####

# labels = ['Reference', 'New', 'Matching']
# values = [data_dict['ref_count'], data_dict['new_count'], data_dict['matching']]

# # Create bar chart
# plt.figure(figsize=(6, 4))
# plt.bar(labels, values, color=['blue', 'orange', 'green'])

# # Add labels and title
# plt.ylabel('Count')
# plt.xlabel('Category')
# plt.title(f"Comparison for '{data_dict['description']}'")
# plt.ylim(0, max(values) * 1.1)

# # Show values on top of bars
# for i, v in enumerate(values):
#     plt.text(i, v + 5000, f"{v:,}", ha='center', fontsize=10)

# # Show the plot
# plt.show()


#### 

# bit_values = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size=1000, p=[0.3, 0.1, 0.1, 0.15, 0.1, 0.1, 0.05, 0.1])

####


ref_qc = nc1['Ref_QF'][:]
new_qc = nc2['DQF'][:]
ref_qc_masked = np.where(valid_mask, ref_qc, np.nan)
new_qc_masked = np.where(valid_mask, new_qc, np.nan)


def compare_qc_flags(ref_qc, new_qc):
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

    bit_start = 0
    bit_length = 3
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

    ref_bits = extract_bits(ref_qc, bit_start, bit_length)
    new_bits = extract_bits(new_qc, bit_start, bit_length)
    
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
    for value in range(bit_length):
        # if value < bit_start or value >= (bit_start + bit_length):
        #     continue

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


# Bit descriptions
bit_labels = {
    0: 'Good',
    1: 'Snow',
    2: 'Heavy aerosol',
    3: 'Fixed aerosol',
    4: 'Cloudy',
    5: 'Large SZA',
    6: 'Large VZA',
    7: 'Bad L1b'
}

# Create histogram
plt.figure(figsize=(8, 5))
plt.hist((diff, ref_valid,new_valid), bins=np.arange(-0.5, 8.5, 1), alpha=0.7, color='blue', edgecolor='black')

# Labels and ticks
plt.xticks(range(8), [bit_labels[i] for i in range(8)], rotation=45, ha='right')
plt.xlabel("Bit Indicator")
plt.ylabel("Frequency")
plt.title("Histogram of Quality Score Bits")

plt.show()