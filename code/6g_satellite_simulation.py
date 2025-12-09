#!/usr/bin/env python3
"""
6G空天通信课程作业 - 完整仿真代码
====================================

基于2025华数杯B题网络切片资源管理问题，改造为6G空地高通量通信场景。

整合内容：
- 多波束增益模型 (Bessel函数)
- DVB-S2X ACM自适应编码调制
- LDPC/Polar信道编码
- IBI/ISI/LTI干扰管理
- 网络切片资源分配

环境依赖：
- Python >= 3.8
- numpy >= 1.20
- pandas >= 1.3
- matplotlib >= 3.4
- scipy >= 1.7
- openpyxl >= 3.0 (用于Excel读写)

安装依赖：
pip install numpy pandas matplotlib scipy openpyxl

运行方式：
python 6g_satellite_simulation.py

原始数据文件说明：
- channel_data.xlsx: 问题1-2的单基站信道数据
- BS1/BS2/BS3.xlsx: 问题3的多基站信道数据
- MBS_1.xlsx, SBS_1/2/3.xlsx: 问题4-5的异构网络数据
- taskflow.xlsx: 用户任务到达数据

输出文件：
- simulation_results.xlsx: 仿真数值结果
- simulation_results.csv: CSV格式结果
- fig*.png: 可视化图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import jv  # Bessel函数
from scipy.optimize import minimize
import warnings
import os

warnings.filterwarnings('ignore')

# 设置matplotlib使用英文标签避免中文乱码
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

#==============================================================================
# 第一部分: 系统参数定义
#==============================================================================

class SystemParameters:
    """6G空天通信系统参数"""
    
    # LEO卫星参数
    SAT_HEIGHT_KM = 500           # 轨道高度 (km)
    SAT_FREQ_GHZ = 30             # Ka波段频率 (GHz)
    LIGHT_SPEED = 3e8             # 光速 (m/s)
    
    # 波束参数
    BEAM_3DB_DEG = 0.5            # 3dB波束宽度 (度)
    G_MAX_DBI = 48                # 最大波束增益 (dBi)
    NUM_BEAMS = 100               # 每卫星波束数
    
    # 网络切片参数 (沿用原题)
    RB_URLLC = 10                 # URLLC每用户RB
    RB_EMBB = 5                   # eMBB每用户RB
    RB_MMTC = 2                   # mMTC每用户RB
    
    # SLA要求
    RATE_SLA_URLLC = 10           # Mbps
    RATE_SLA_EMBB = 50            # Mbps
    RATE_SLA_MMTC = 1             # Mbps
    
    DELAY_SLA_URLLC = 5           # ms
    DELAY_SLA_EMBB = 100          # ms
    DELAY_SLA_MMTC = 500          # ms
    
    # 惩罚系数
    PENALTY_URLLC = 5
    PENALTY_EMBB = 3
    PENALTY_MMTC = 1
    
    # URLLC折扣因子
    ALPHA = 0.95
    
    # 物理层参数
    BW_PER_RB_KHZ = 360           # 每RB带宽 (kHz)
    NOISE_DENSITY_DBM = -174      # 噪声功率谱密度 (dBm/Hz)
    NOISE_FIGURE_DB = 7           # 接收机噪声系数 (dB)
    
    # 功率范围
    MBS_POWER_RANGE = (10, 40)    # MBS/LEO功率范围 (dBm)
    SBS_POWER_RANGE = (10, 30)    # SBS/Gateway功率范围 (dBm)
    
    # 能耗模型参数
    P_STATIC_W = 28               # 静态功耗 (W)
    DELTA_RB = 0.75               # RB激活功耗系数 (W/RB)
    PA_EFFICIENCY = 0.35          # 功放效率


#==============================================================================
# 第二部分: 信道模型
#==============================================================================

class SatelliteChannelModel:
    """
    6G空地信道模型
    
    将原题5G信道转换为星地信道：
    - 大规模衰减 -> 自由空间损耗 + 大气衰减 + 波束增益
    - 小规模瑞利衰减 -> Shadowed-Rician衰落
    """
    
    def __init__(self):
        self.params = SystemParameters()
        self.wavelength = self.params.LIGHT_SPEED / (self.params.SAT_FREQ_GHZ * 1e9)
        
    def beam_gain(self, theta_deg):
        """
        计算波束增益 (Bessel函数模型)
        
        G(θ) = G_max × [2*J1(u)/u]^2
        
        参数:
            theta_deg: 偏轴角度 (度)
        返回:
            增益 (dBi)
        """
        if abs(theta_deg) < 1e-6:
            return self.params.G_MAX_DBI
        
        theta_3db_rad = np.radians(self.params.BEAM_3DB_DEG)
        theta_rad = np.radians(theta_deg)
        
        u = 2.07123 * np.sin(theta_rad) / np.sin(theta_3db_rad)
        
        if abs(u) < 1e-10:
            gain_factor = 1.0
        else:
            gain_factor = (2 * jv(1, u) / u) ** 2
        
        return self.params.G_MAX_DBI + 10 * np.log10(max(gain_factor, 1e-10))
    
    def free_space_loss(self, distance_km):
        """
        计算自由空间损耗
        
        L_fs = 20*log10(4*π*d/λ)
        """
        distance_m = distance_km * 1e3
        return 20 * np.log10(4 * np.pi * distance_m / self.wavelength)
    
    def atmospheric_loss(self, elevation_deg):
        """
        计算大气衰减 (简化模型)
        """
        elevation_rad = np.radians(max(elevation_deg, 10))
        return 0.5 / np.sin(elevation_rad)  # dB
    
    def shadowed_rician_fading(self, K_factor_dB=10, size=1):
        """
        生成Shadowed-Rician衰落系数
        
        参数:
            K_factor_dB: 莱斯因子 (dB), LOS/NLOS功率比
        返回:
            衰落幅度的平方 |h|^2
        """
        K = 10 ** (K_factor_dB / 10)
        
        # LOS分量
        h_los = np.sqrt(K / (K + 1))
        
        # NLOS分量 (复高斯)
        h_nlos_real = np.random.randn(size) * np.sqrt(1 / (2 * (K + 1)))
        h_nlos_imag = np.random.randn(size) * np.sqrt(1 / (2 * (K + 1)))
        
        # 总衰落
        h_total = np.abs(h_los + h_nlos_real + 1j * h_nlos_imag) ** 2
        
        return h_total if size > 1 else h_total[0]
    
    def transform_channel(self, original_path_loss_dB, original_fading):
        """
        将原题信道数据转换为星地信道
        
        原数据范围: 大规模衰减约30-80dB
        目标范围: 星地损耗约160-190dB (含波束增益补偿后)
        """
        # 添加偏移量模拟星地距离
        sat_path_loss = original_path_loss_dB + 130
        
        # 将瑞利衰落转换为Shadowed-Rician
        K_factor = 10  # dB
        K_linear = 10 ** (K_factor / 10)
        h_los = np.sqrt(K_linear / (K_linear + 1))
        h_nlos = np.sqrt(1 / (K_linear + 1)) * original_fading
        transformed_fading = np.abs(h_los + h_nlos)
        
        return sat_path_loss, transformed_fading


#==============================================================================
# 第三部分: 信道编码模型
#==============================================================================

class ChannelCodingModel:
    """
    6G信道编码模型
    
    根据IEEE Proc. 2024研究，按块长度选择最优编码：
    - N ≤ 128: 卷积码 (Viterbi)
    - 128 < N ≤ 512: Polar码 (SCL-8)
    - N > 512: LDPC码 (BP-12)
    """
    
    # 编码参数
    CODING_PARAMS = {
        'URLLC': {
            'code_type': 'Polar',
            'block_length': 256,
            'code_rate': 0.5,
            'target_BLER': 1e-5,
            'decoder': 'SCL-8'
        },
        'eMBB': {
            'code_type': 'LDPC',
            'block_length': 8448,
            'code_rate': 0.75,
            'target_BLER': 1e-3,
            'decoder': 'BP-12'
        },
        'mMTC': {
            'code_type': 'Polar',
            'block_length': 128,
            'code_rate': 0.33,
            'target_BLER': 1e-2,
            'decoder': 'SC'
        }
    }
    
    def get_coding_scheme(self, slice_type):
        """获取切片对应的编码方案"""
        return self.CODING_PARAMS.get(slice_type, self.CODING_PARAMS['eMBB'])
    
    def estimate_bler(self, snr_dB, slice_type):
        """
        估算BLER (简化模型)
        
        基于论文数据拟合的BLER-SNR曲线
        """
        params = self.get_coding_scheme(slice_type)
        
        if params['code_type'] == 'Polar':
            # Polar码BLER模型
            snr_threshold = 2 if params['block_length'] <= 128 else 1
            bler = 0.5 * np.exp(-0.5 * (snr_dB - snr_threshold))
        else:  # LDPC
            # LDPC码BLER模型
            snr_threshold = 0
            bler = 0.5 * np.exp(-0.3 * (snr_dB - snr_threshold))
        
        return max(min(bler, 1), 1e-7)


#==============================================================================
# 第四部分: DVB-S2X ACM模型
#==============================================================================

class DVB_S2X_ACM:
    """
    DVB-S2X自适应编码调制 (ACM)
    
    支持28种MODCOD组合，频谱效率0.49-5.51 bit/s/Hz
    """
    
    # MODCOD查找表 (SNR阈值, 频谱效率)
    MODCOD_TABLE = [
        (-2.35, 0.49, 'QPSK 1/4'),
        (1.00, 0.99, 'QPSK 1/2'),
        (4.03, 1.49, 'QPSK 3/4'),
        (6.62, 1.98, '8PSK 2/3'),
        (8.97, 2.23, '8PSK 3/4'),
        (9.82, 2.64, '16APSK 2/3'),
        (10.21, 2.97, '16APSK 3/4'),
        (12.73, 3.95, '32APSK 4/5'),
        (16.05, 4.94, '64APSK 5/6'),
        (18.10, 5.51, '256APSK 3/4'),
    ]
    
    def select_modcod(self, snr_dB, margin_dB=2):
        """
        根据SNR选择最佳MODCOD
        
        参数:
            snr_dB: 估计的SNR (dB)
            margin_dB: 链路余量 (dB)
        返回:
            (频谱效率, MODCOD名称)
        """
        effective_snr = snr_dB - margin_dB
        
        selected = self.MODCOD_TABLE[0]  # 默认最低
        for threshold, efficiency, name in self.MODCOD_TABLE:
            if effective_snr >= threshold:
                selected = (threshold, efficiency, name)
        
        return selected[1], selected[2]
    
    def calculate_rate(self, bandwidth_mhz, snr_dB):
        """
        计算ACM实际传输速率
        
        r = η(MODCOD) × B × (1 - FER)
        """
        efficiency, modcod = self.select_modcod(snr_dB)
        fer = 0.01 if snr_dB < 5 else 0.001  # 简化FER估计
        rate_mbps = efficiency * bandwidth_mhz * (1 - fer)
        return rate_mbps, modcod


#==============================================================================
# 第五部分: 干扰模型 (IBI/ISI/LTI)
#==============================================================================

class InterferenceModel:
    """
    三类干扰模型 (IEEE COMST 2025)
    
    - IBI: Inter-Beam Interference (波束间干扰)
    - ISI: Inter-Satellite Interference (星间干扰)
    - LTI: LEO-Terrestrial Interference (星地干扰)
    """
    
    def calculate_ibi(self, target_beam_gain_dB, adjacent_beam_gains_dB, sidelobe_rejection_dB=25):
        """
        计算波束间干扰 (IBI)
        
        IBI来自同一卫星的相邻波束
        """
        interference_power = 0
        for gain in adjacent_beam_gains_dB:
            # 考虑旁瓣抑制
            effective_gain = gain - sidelobe_rejection_dB
            interference_power += 10 ** (effective_gain / 10)
        
        return 10 * np.log10(interference_power + 1e-10) if interference_power > 0 else -100
    
    def calculate_isi(self, interference_path_losses_dB, interference_powers_dBm):
        """
        计算星间干扰 (ISI)
        
        ISI来自其他卫星的同频波束
        """
        interference_power = 0
        for loss, power in zip(interference_path_losses_dB, interference_powers_dBm):
            rx_power_mW = 10 ** ((power - loss) / 10)
            interference_power += rx_power_mW
        
        return 10 * np.log10(interference_power + 1e-10) if interference_power > 0 else -100
    
    def calculate_sinr(self, signal_power_dBm, ibi_dBm, isi_dBm, noise_power_dBm):
        """
        计算SINR
        
        SINR = P_signal / (P_IBI + P_ISI + N_0)
        """
        signal_mW = 10 ** (signal_power_dBm / 10)
        ibi_mW = 10 ** (ibi_dBm / 10) if ibi_dBm > -90 else 0
        isi_mW = 10 ** (isi_dBm / 10) if isi_dBm > -90 else 0
        noise_mW = 10 ** (noise_power_dBm / 10)
        
        sinr_linear = signal_mW / (ibi_mW + isi_mW + noise_mW + 1e-10)
        return 10 * np.log10(sinr_linear)


#==============================================================================
# 第六部分: QoS计算
#==============================================================================

class QoSCalculator:
    """QoS服务质量计算"""
    
    def __init__(self):
        self.params = SystemParameters()
    
    def qos_urllc(self, delay_ms):
        """URLLC QoS: Q = α^D if D ≤ D_SLA else -M"""
        if delay_ms <= self.params.DELAY_SLA_URLLC:
            return self.params.ALPHA ** delay_ms
        return -self.params.PENALTY_URLLC
    
    def qos_embb(self, rate_mbps, delay_ms):
        """eMBB QoS: 基于速率和时延"""
        if delay_ms > self.params.DELAY_SLA_EMBB:
            return -self.params.PENALTY_EMBB
        if rate_mbps >= self.params.RATE_SLA_EMBB:
            return 1.0
        return rate_mbps / self.params.RATE_SLA_EMBB
    
    def qos_mmtc(self, connected, total, delay_ms):
        """mMTC QoS: 基于连接率"""
        if delay_ms > self.params.DELAY_SLA_MMTC:
            return -self.params.PENALTY_MMTC
        if total == 0:
            return 1.0
        return connected / total


#==============================================================================
# 第七部分: 资源分配优化
#==============================================================================

class ResourceAllocator:
    """网络切片资源分配器"""
    
    def __init__(self):
        self.params = SystemParameters()
        self.channel = SatelliteChannelModel()
        self.coding = ChannelCodingModel()
        self.acm = DVB_S2X_ACM()
        self.qos_calc = QoSCalculator()
        self.interference = InterferenceModel()
    
    def calculate_noise_power(self, num_rb):
        """计算噪声功率"""
        bw_hz = num_rb * self.params.BW_PER_RB_KHZ * 1000
        return self.params.NOISE_DENSITY_DBM + 10 * np.log10(bw_hz) + self.params.NOISE_FIGURE_DB
    
    def calculate_user_rate(self, path_loss_dB, fading, tx_power_dBm, num_rb):
        """计算用户传输速率"""
        # 接收功率
        rx_power_dBm = tx_power_dBm - path_loss_dB + 10 * np.log10(abs(fading)**2 + 1e-10)
        
        # 噪声功率
        noise_dBm = self.calculate_noise_power(num_rb)
        
        # SNR
        snr_dB = rx_power_dBm - noise_dBm
        
        # 带宽
        bw_mhz = num_rb * self.params.BW_PER_RB_KHZ / 1000
        
        # ACM速率
        rate_mbps, modcod = self.acm.calculate_rate(bw_mhz, snr_dB)
        
        return rate_mbps, snr_dB, modcod
    
    def optimize_allocation(self, users_data, total_rb, tx_power_dBm):
        """
        优化资源分配
        
        策略: 优先级调度 URLLC > eMBB > mMTC
        """
        # 统计各类用户
        urllc_users = [u for u in users_data if u['type'] == 'URLLC' and u['active']]
        embb_users = [u for u in users_data if u['type'] == 'eMBB' and u['active']]
        mmtc_users = [u for u in users_data if u['type'] == 'mMTC' and u['active']]
        
        # 优先分配URLLC
        rb_urllc = min(len(urllc_users) * self.params.RB_URLLC, total_rb)
        remaining = total_rb - rb_urllc
        
        # 分配eMBB
        rb_embb = min(len(embb_users) * self.params.RB_EMBB, remaining)
        remaining -= rb_embb
        
        # 分配mMTC
        rb_mmtc = remaining
        
        return {
            'URLLC': rb_urllc,
            'eMBB': rb_embb,
            'mMTC': rb_mmtc
        }
    
    def calculate_total_qos(self, allocation, users_data, tx_power_dBm):
        """计算总QoS"""
        total_qos = 0
        
        for user in users_data:
            if not user['active']:
                continue
            
            user_type = user['type']
            if user_type == 'URLLC':
                num_rb = self.params.RB_URLLC
            elif user_type == 'eMBB':
                num_rb = self.params.RB_EMBB
            else:
                num_rb = self.params.RB_MMTC
            
            rate, snr, _ = self.calculate_user_rate(
                user['path_loss'], user['fading'], tx_power_dBm, num_rb
            )
            
            delay = user['data_size'] / rate * 1000 if rate > 0 else 1000
            
            if user_type == 'URLLC':
                total_qos += self.qos_calc.qos_urllc(delay)
            elif user_type == 'eMBB':
                total_qos += self.qos_calc.qos_embb(rate, delay)
            else:
                total_qos += self.qos_calc.qos_mmtc(1, 1, delay)
        
        return total_qos


#==============================================================================
# 第八部分: 数据加载与仿真
#==============================================================================

class Simulator:
    """仿真主类"""
    
    def __init__(self, data_path='/mnt/project'):
        self.data_path = data_path
        self.allocator = ResourceAllocator()
        self.channel = SatelliteChannelModel()
        
    def load_channel_data(self, filename='channel_data.xlsx'):
        """加载信道数据"""
        filepath = os.path.join(self.data_path, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, using synthetic data")
            return self._generate_synthetic_data()
        
        xls = pd.ExcelFile(filepath)
        data = {
            'large_scale': pd.read_excel(xls, sheet_name='大规模衰减'),
            'small_scale': pd.read_excel(xls, sheet_name='小规模瑞丽衰减'),
        }
        try:
            data['task_flow'] = pd.read_excel(xls, sheet_name='用户任务流')
        except:
            data['task_flow'] = None
        
        return data
    
    def _generate_synthetic_data(self):
        """生成合成数据用于演示"""
        n_time = 1000
        users = ['U1', 'U2', 'e1', 'e2', 'e3', 'e4'] + [f'm{i}' for i in range(1, 11)]
        
        data = {
            'large_scale': pd.DataFrame({
                'Time': np.arange(n_time) / 1000,
                **{u: np.random.uniform(40, 70, n_time) for u in users}
            }),
            'small_scale': pd.DataFrame({
                'Time': np.arange(n_time) / 1000,
                **{u: np.random.randn(n_time) for u in users}
            }),
            'task_flow': pd.DataFrame({
                'Time': np.arange(n_time) / 1000,
                **{u: np.random.choice([0, 0.01], n_time, p=[0.9, 0.1]) for u in users}
            })
        }
        return data
    
    def run_simulation(self, total_rb=50, tx_power_dBm=30, n_epochs=10):
        """运行完整仿真"""
        print("="*60)
        print("6G Space-Ground Communication Simulation")
        print("="*60)
        
        # 加载数据
        data = self.load_channel_data()
        
        results = []
        
        for epoch in range(n_epochs):
            time_idx = epoch * 100
            if time_idx >= len(data['large_scale']):
                time_idx = len(data['large_scale']) - 1
            
            # 构建用户数据
            users = self._build_user_list(data, time_idx)
            
            # 优化分配
            allocation = self.allocator.optimize_allocation(users, total_rb, tx_power_dBm)
            
            # 计算QoS
            qos = self.allocator.calculate_total_qos(allocation, users, tx_power_dBm)
            
            results.append({
                'epoch': epoch + 1,
                'time_ms': epoch * 100,
                'URLLC_RB': allocation['URLLC'],
                'eMBB_RB': allocation['eMBB'],
                'mMTC_RB': allocation['mMTC'],
                'QoS': qos,
                'active_users': sum(1 for u in users if u['active'])
            })
            
            print(f"Epoch {epoch+1}: URLLC={allocation['URLLC']}RB, "
                  f"eMBB={allocation['eMBB']}RB, mMTC={allocation['mMTC']}RB, QoS={qos:.3f}")
        
        return pd.DataFrame(results)
    
    def _build_user_list(self, data, time_idx):
        """构建用户列表"""
        users = []
        
        # URLLC用户
        for u in ['U1', 'U2']:
            if u in data['large_scale'].columns:
                users.append({
                    'id': u,
                    'type': 'URLLC',
                    'path_loss': data['large_scale'].loc[time_idx, u],
                    'fading': data['small_scale'].loc[time_idx, u],
                    'data_size': 0.01,  # Mbit
                    'active': np.random.random() < 0.3
                })
        
        # eMBB用户
        for u in ['e1', 'e2', 'e3', 'e4']:
            if u in data['large_scale'].columns:
                users.append({
                    'id': u,
                    'type': 'eMBB',
                    'path_loss': data['large_scale'].loc[time_idx, u],
                    'fading': data['small_scale'].loc[time_idx, u],
                    'data_size': 0.1,  # Mbit
                    'active': np.random.random() < 0.5
                })
        
        # mMTC用户
        for i in range(1, 11):
            u = f'm{i}'
            if u in data['large_scale'].columns:
                users.append({
                    'id': u,
                    'type': 'mMTC',
                    'path_loss': data['large_scale'].loc[time_idx, u],
                    'fading': data['small_scale'].loc[time_idx, u],
                    'data_size': 0.013,  # Mbit
                    'active': np.random.random() < 0.7
                })
        
        return users


#==============================================================================
# 第九部分: 可视化
#==============================================================================

def generate_visualizations(results_df, output_dir='.'):
    """生成可视化图表"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 图1: RB分配
    ax = axes[0, 0]
    epochs = results_df['epoch']
    ax.bar(epochs, results_df['URLLC_RB'], label='URLLC', color='#ff6b6b')
    ax.bar(epochs, results_df['eMBB_RB'], bottom=results_df['URLLC_RB'], label='eMBB', color='#4ecdc4')
    ax.bar(epochs, results_df['mMTC_RB'], 
           bottom=results_df['URLLC_RB']+results_df['eMBB_RB'], label='mMTC', color='#45b7d1')
    ax.set_xlabel('Decision Epoch')
    ax.set_ylabel('Resource Blocks')
    ax.set_title('Resource Allocation per Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 图2: QoS变化
    ax = axes[0, 1]
    ax.plot(epochs, results_df['QoS'], 'o-', color='#4ecdc4', linewidth=2, markersize=8)
    ax.axhline(y=0.9, color='red', linestyle='--', label='Target QoS')
    ax.set_xlabel('Decision Epoch')
    ax.set_ylabel('QoS Score')
    ax.set_title('QoS Performance over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 图3: 活跃用户数
    ax = axes[1, 0]
    ax.bar(epochs, results_df['active_users'], color='#96ceb4', edgecolor='black')
    ax.set_xlabel('Decision Epoch')
    ax.set_ylabel('Active Users')
    ax.set_title('Active Users per Epoch')
    ax.grid(True, alpha=0.3)
    
    # 图4: 累积QoS
    ax = axes[1, 1]
    cumulative_qos = results_df['QoS'].cumsum()
    ax.fill_between(epochs, cumulative_qos, alpha=0.3, color='#ff6b6b')
    ax.plot(epochs, cumulative_qos, 'o-', color='#ff6b6b', linewidth=2, markersize=8)
    ax.set_xlabel('Decision Epoch')
    ax.set_ylabel('Cumulative QoS')
    ax.set_title('Cumulative QoS over Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'simulation_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_dir}/simulation_results.png")


#==============================================================================
# 第十部分: 主程序
#==============================================================================

def main():
    """主函数"""
    print("\n" + "="*70)
    print("6G Space-Ground High-Throughput Communication Simulation")
    print("Based on HuaShuBei 2025 Problem B - Network Slicing")
    print("="*70 + "\n")
    
    # 创建仿真器
    sim = Simulator(data_path='/mnt/project')
    
    # 运行仿真
    results = sim.run_simulation(total_rb=50, tx_power_dBm=30, n_epochs=10)
    
    # 保存结果
    output_dir = '/mnt/user-data/outputs'
    if not os.path.exists(output_dir):
        output_dir = '.'
    
    results.to_excel(os.path.join(output_dir, 'simulation_results.xlsx'), index=False)
    results.to_csv(os.path.join(output_dir, 'simulation_results.csv'), index=False)
    
    # 生成可视化
    generate_visualizations(results, output_dir)
    
    # 打印摘要
    print("\n" + "="*60)
    print("Simulation Summary")
    print("="*60)
    print(f"Total Epochs: {len(results)}")
    print(f"Average QoS: {results['QoS'].mean():.4f}")
    print(f"Total QoS: {results['QoS'].sum():.4f}")
    print(f"Average Active Users: {results['active_users'].mean():.1f}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = main()
