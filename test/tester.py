import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_comparison(ref_data, new_data, diff_data, title, output_dir):
    """在1x3子图中绘制比较结果"""
    fig = plt.figure(figsize=(24, 8))
    
    # 设置投影参数
    semi_major = 6378137.0
    semi_minor = 6356752.31414
    longitude_of_projection_origin = -75.0
    perspective_point_height = 3.5786023E7
    globe = ccrs.Globe(ellipse='sphere', semimajor_axis=semi_major, semiminor_axis=semi_minor)
    projection = ccrs.Geostationary(central_longitude=longitude_of_projection_origin, 
                                  satellite_height=perspective_point_height, globe=globe)
    
    titles = ['Reference', 'New', 'Difference']
    data_list = [ref_data, new_data, diff_data]
    cmaps = ['viridis', 'viridis', 'seismic']
    
    # 计算数据范围
    valid_min = min(np.nanmin(ref_data), np.nanmin(new_data))
    valid_max = max(np.nanmax(ref_data), np.nanmax(new_data))
    diff_range = max(abs(np.nanmin(diff_data)), abs(np.nanmax(diff_data)))
    
    ranges = [(valid_min, valid_max), (valid_min, valid_max), (-diff_range, diff_range)]
    
    for idx in range(3):
        ax = fig.add_subplot(1, 3, idx+1, projection=projection)
        
        img = ax.imshow(data_list[idx], origin='upper', transform=projection,
                       extent=(-5434894.8823, 5434894.8823, -5434894.8823, 5434894.8823),
                       cmap=cmaps[idx], vmin=ranges[idx][0], vmax=ranges[idx][1])
        
        ax.gridlines(color='gray', alpha=0.5)
        ax.coastlines(resolution='50m', color='black', linestyle='--')
        
        plt.colorbar(img, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
        ax.set_title(f'{titles[idx]} - {title}', fontsize=14, pad=20)
    
    plt.tight_layout()
    output_name = os.path.join(output_dir, f'BRF_Comparison_{title.replace(" ", "_")}.png')
    fig.savefig(output_name, bbox_inches='tight')
    plt.close()

def extract_bits(qc_data, start_bit, num_bits):
    """从QC标志中提取指定的位"""
    # 将nan值替换为255，并转换为整数类型
    qc_int = np.nan_to_num(qc_data, nan=255).astype(np.uint8)
    mask = ((1 << num_bits) - 1) << start_bit
    extracted_bits = (qc_int & mask) >> start_bit
    
    # 将原始数据中的nan位置在结果中也设为nan
    result = extracted_bits.astype(float)
    result[np.isnan(qc_data)] = np.nan
    return result

def plot_qc_comparison(ref_qc, new_qc, flag_name, flag_info, output_dir):
    """为单个QC标志位绘制比较图"""
    # 提取当前标志位的数据
    ref_bits = extract_bits(ref_qc, flag_info['start'], flag_info['bits'])
    new_bits = extract_bits(new_qc, flag_info['start'], flag_info['bits'])
    
    # 计算差异，保持nan值
    diff = ref_bits - new_bits
    
    # 创建图像
    fig = plt.figure(figsize=(24, 8))
    
    # 设置投影参数
    semi_major = 6378137.0
    semi_minor = 6356752.31414
    longitude_of_projection_origin = -75.0
    perspective_point_height = 3.5786023E7
    globe = ccrs.Globe(ellipse='sphere', semimajor_axis=semi_major, semiminor_axis=semi_minor)
    projection = ccrs.Geostationary(central_longitude=longitude_of_projection_origin, 
                                  satellite_height=perspective_point_height, globe=globe)
    
    titles = ['Reference', 'New', 'Difference']
    data_list = [ref_bits, new_bits, diff]
    
    # 为QC标志设置合适的colormap
    max_value = (1 << flag_info['bits']) - 1
    cmaps = ['viridis', 'viridis', 'seismic']
    ranges = [(0, max_value), (0, max_value), (-max_value, max_value)]
    
    for idx in range(3):
        ax = fig.add_subplot(1, 3, idx+1, projection=projection)
        
        img = ax.imshow(data_list[idx], origin='upper', transform=projection,
                       extent=(-5434894.8823, 5434894.8823, -5434894.8823, 5434894.8823),
                       cmap=cmaps[idx], vmin=ranges[idx][0], vmax=ranges[idx][1])
        
        ax.gridlines(color='gray', alpha=0.5)
        ax.coastlines(resolution='50m', color='black', linestyle='--')
        
        # 添加colorbar
        cbar = plt.colorbar(img, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
        
        # 为前两个图（参考和新数据）添加值的说明
        if idx < 2:
            cbar.set_ticks(range(max_value + 1))
            cbar.set_ticklabels([f"{i}\n({flag_info['values'][i]})" for i in range(max_value + 1)])
        
        ax.set_title(f'{titles[idx]} - {flag_name.replace("_", " ").title()}', fontsize=14, pad=20)
    
    plt.tight_layout()
    output_name = os.path.join(output_dir, f'BRF_QC_Comparison_{flag_name}.png')
    fig.savefig(output_name, bbox_inches='tight')
    plt.close()

def analyze_aod_availability_changes(ref_bits, new_bits, valid_mask):
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

def compare_qc_flags(ref_qc, new_qc, output_dir):
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

    
    # 为每个标志位绘制比较图
    for flag_name, flag_info in brf_dqf_bits.items():
        plot_qc_comparison(ref_qc, new_qc, flag_name, flag_info, output_dir)
    
    results = {}
    total_valid_pixels = np.sum(~np.isnan(ref_qc))
    
    # 分析每个标志位
    for flag_name, flag_info in brf_dqf_bits.items():
        ref_bits = extract_bits(ref_qc, flag_info['start'], flag_info['bits'])
        new_bits = extract_bits(new_qc, flag_info['start'], flag_info['bits'])
        
        valid_pixels = ~np.isnan(ref_bits)
        
        # 如果是aod_availability标记，进行采样分析
        if flag_name == 'aod_availability':
            total_diff = analyze_aod_availability_changes(ref_bits, new_bits, valid_pixels)
        
        # 计算每个可能值的统计
        value_stats = {}
        valid_pixels = ~np.isnan(ref_bits)
        
        # 计算总体匹配统计
        matching_pixels = np.sum((ref_bits == new_bits) & valid_pixels)
        different_pixels = np.sum((ref_bits != new_bits) & valid_pixels)
        
        print(f"\n{flag_name.replace('_', ' ').title()}:")
        print(f"  Matching pixels: {matching_pixels}")
        print(f"  Different pixels: {different_pixels}")
        print(f"  Matching percentage: {(matching_pixels / total_valid_pixels * 100):.2f}%")
        
        # 详细的值分布统计
        print("  Value distribution:")
        for value in flag_info['values'].keys():
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

def compare_brf_files(file1, file2, output_dir):
    """比较两个BRF文件"""
    # 读取文件
    nc1 = Dataset(file1, 'r')  # Reference file
    nc2 = Dataset(file2, 'r')  # New file
    
    # 比较反射率
    results = {}
    
    # 通道映射
    channels = {
        1: 'Ch1_Ref',
        2: 'Ch2_Ref',
        3: 'Ch3_Ref',
        5: 'Ch5_Ref',
        6: 'Ch6_Ref'
    }
    
    # 比较每个通道的反射率
    for band, ref_var in channels.items():
        # 读取反射率数据
        ref_data = nc1[ref_var][:]
        new_data = nc2[f'BRF{band}'][:]
        
        # 创建掩码处理缺失值
        ref_mask = ((ref_data != nc1[ref_var]._FillValue) & 
                    (ref_data >= 0) & 
                    (ref_data <= 1))
        new_mask = ((new_data != nc2[f'BRF{band}']._FillValue) & 
                    (new_data >= 0) & 
                    (new_data <= 1))
        valid_mask = ref_mask & new_mask
        
        # 计算有效数据的统计信息
        ref_valid = ref_data[valid_mask]
        new_valid = new_data[valid_mask]
        diff = np.where(valid_mask, ref_data - new_data, np.nan)
        
        # 计算统计数据
        stats = {
            'ref_mean': np.mean(ref_valid),
            'new_mean': np.mean(new_valid),
            'mean_diff': np.nanmean(diff),
            'std_diff': np.nanstd(diff),
            'max_diff': np.nanmax(np.abs(diff)),
            'valid_pixels': np.sum(valid_mask),
            'relative_diff_percent': (np.nanmean(np.abs(diff)) / np.nanmean(np.abs(ref_valid))) * 100
        }
        
        # 绘制比较图
        plot_comparison(
            ref_data,
            new_data,
            diff,
            f'Band {band}',
            output_dir
        )
        
        results[f'Band{band}'] = stats
    
    # 比较QC标记 - 使用反射率的有效掩码
    ref_qc = nc1['Ref_QF'][:]
    new_qc = nc2['DQF'][:]
    
    # 将无效区域的QC设为nan
    ref_qc_masked = np.where(valid_mask, ref_qc, np.nan)
    new_qc_masked = np.where(valid_mask, new_qc, np.nan)
    
    # 详细的QC比较
    qc_results = compare_qc_flags(ref_qc_masked, new_qc_masked, output_dir)
    
    all_results = {
        'reflectance': results,
        'qc': qc_results
    }
    
    nc1.close()
    nc2.close()
    return all_results

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

def main():
    # 创建输出目录
    output_dir = 'results/brf_analysis'
    ensure_dir(output_dir)
    
    # 文件路径
    ref_file = '/home/brian/research/test/G16_ABI_FD_2025011_1900_00_LAND_SFC_REFLECTANCE_EN (1).nc'
    new_file = '/home/brian/research/test/DR_ABI-L2-BRFF-M6_G16_s20250111900205_e20250111909513_c20250111914544 (1).nc'
    
    # 比较文件
    print("Comparing BRF files...")
    results = compare_brf_files(ref_file, new_file, output_dir)
    
    # 保存结果
    save_results_to_file(results, output_dir)
    
    print("\nResults have been saved to the 'results/brf_analysis' folder.")
    print("Check 'brf_comparison_results.txt' for detailed results.")

if __name__ == "__main__":
    main() 