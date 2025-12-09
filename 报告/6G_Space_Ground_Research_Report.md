# 面向6G的全域覆盖：空天通信深度研究报告

## 基于网络切片无线资源管理的空地高通量通信系统设计

---

## 一、研究背景与问题重构

### 1.1 原建模题核心架构分析

原题目为5G网络切片无线资源管理，核心要素：

| 原题要素 | 技术描述 | 数据特征 |
|---------|---------|---------|
| **宏基站(MBS)** | 大范围覆盖，100个RB，功率10-40dBm | MBS_1.xlsx: 70用户信道数据 |
| **微基站(SBS)** | 边缘增强，50个RB，功率10-30dBm | SBS_1/2/3.xlsx: 三站协同 |
| **网络切片** | URLLC/eMBB/mMTC三类切片 | 10+20+40=70用户 |
| **频率复用干扰** | 微基站间同频干扰 | 信干噪比SINR计算 |
| **资源调度** | 时域100ms周期决策 | 1000ms任务流数据 |

### 1.2 6G空天通信场景重构

**核心改造思路**：将地面异构网络映射为空地异构网络

| 原5G架构 | 6G空天架构 | 技术对应 |
|---------|-----------|---------|
| 宏基站(MBS) | **LEO卫星/HAPS高空平台** | 全域覆盖层 |
| 微基站(SBS) | **地面网关站/无人机中继** | 边缘增强层 |
| 基站位置固定 | **高动态星座轨道运动** | 时变拓扑 |
| 频率复用干扰 | **星间干扰/多波束干扰** | 空间干扰协调 |
| OFDMA | **OTFS/多波束OFDMA** | 抗多普勒调制 |

---

## 二、6G空地集成网络架构设计

### 2.1 系统架构：三层异构网络

```
┌─────────────────────────────────────────────────────────────┐
│                    空间层 (Space Layer)                      │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐      │
│  │  LEO-1  │   │  LEO-2  │   │  LEO-3  │   │  LEO-4  │      │
│  │ 500km   │   │ 500km   │   │ 500km   │   │ 500km   │      │
│  │100 Beams│   │100 Beams│   │100 Beams│   │100 Beams│      │
│  └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘      │
│       │             │             │             │            │
│       └─────────────┴──────┬──────┴─────────────┘            │
│                            │ ISL (星间链路)                   │
└────────────────────────────┼────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────┐
│                    空中层 (Aerial Layer)                     │
│       ┌────────────────────┴────────────────────┐           │
│       │            HAPS (20km)                  │           │
│       │     高空平台站 - 区域协调控制            │           │
│       └─────────────────┬───────────────────────┘           │
│                         │                                    │
│    ┌────────┐     ┌─────┴─────┐     ┌────────┐              │
│    │ UAV-1  │     │  UAV-2    │     │ UAV-3  │              │
│    │  中继  │     │   中继    │     │  中继  │              │
│    └───┬────┘     └─────┬─────┘     └────┬───┘              │
└────────┼────────────────┼────────────────┼──────────────────┘
         │                │                │
┌────────┼────────────────┼────────────────┼──────────────────┐
│        │     地面层 (Terrestrial Layer)  │                  │
│   ┌────┴────┐     ┌─────┴─────┐    ┌─────┴────┐             │
│   │ 网关站-1│     │ 网关站-2  │    │ 网关站-3 │             │
│   │(对应SBS)│     │ (对应SBS) │    │(对应SBS) │             │
│   └────┬────┘     └─────┬─────┘    └─────┬────┘             │
│        │                │                │                   │
│    ┌───┴───┐        ┌───┴───┐        ┌───┴───┐              │
│    │URLLC  │        │eMBB   │        │mMTC   │              │
│    │用户群 │        │用户群 │        │用户群 │              │
│    └───────┘        └───────┘        └───────┘              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 参数映射表

| 原题参数 | 6G空天参数 | 物理意义 | 典型值 |
|---------|-----------|---------|--------|
| MBS位置(0,0) | LEO轨道高度 | 卫星高度 | 500-1200 km |
| SBS位置(±433, ±250) | 网关站坐标 | 地面站分布 | 间距500-1000 km |
| 大规模衰减φ | 自由空间损耗+大气衰减 | 路径损耗 | 170-190 dB |
| 小规模瑞利衰减h | Shadowed-Rician衰落 | 星地信道 | Loo模型参数 |
| 资源块带宽360kHz | 子载波带宽 | OFDM参数 | 360kHz (Ka波段) |
| 时隙10ms | 波束驻留时间 | Beam Hopping | 10ms |
| 功率10-40dBm | EIRP | 等效全向辐射功率 | 40-60 dBW |

---

## 三、核心技术模块深度解析

### 3.1 多点波束技术 (Multi-Beam Technology)

#### 3.1.1 技术原理

多波束卫星通过相控阵天线产生多个独立波束，每个波束可独立服务不同地理区域。

**波束增益模型**：
```
G(θ) = G_max × [J₁(u)/2u + 36·J₃(u)/u³]²

其中：
- G_max: 波束峰值增益 (典型值45-50 dBi)
- u = 2.07123 × sin(θ)/sin(θ_3dB)
- θ: 偏离波束中心角度
- θ_3dB: 3dB波束宽度
- J₁, J₃: 第一类贝塞尔函数
```

**与原题对应**：
- 原题中每个用户的"大规模衰减"对应**波束增益+路径损耗**
- 原题的用户位置数据可转化为**用户相对波束中心的偏离角**

#### 3.1.2 数据改造方案

将原`channel_data.xlsx`中的大规模衰减改造为：

```python
# 原数据：φ_n,k (dB) - 用户k到基站n的大规模衰减
# 改造后：L_total = L_free + L_atm + G_beam(θ)

def transform_path_loss(original_phi, user_pos, beam_center, sat_height=500):
    """
    将原题大规模衰减转换为星地链路损耗
    """
    # 自由空间损耗 (Ka波段, 30GHz)
    distance = np.sqrt((user_pos[0]-beam_center[0])**2 + 
                       (user_pos[1]-beam_center[1])**2 + 
                       sat_height**2)
    L_free = 20*np.log10(distance*1e3) + 20*np.log10(30e9) - 147.55
    
    # 大气衰减 (典型值)
    L_atm = 0.5 * (sat_height / 500)  # dB
    
    # 波束增益
    theta = np.arctan(np.sqrt((user_pos[0]-beam_center[0])**2 + 
                              (user_pos[1]-beam_center[1])**2) / sat_height)
    G_beam = calculate_beam_gain(theta, G_max=48)
    
    return L_free + L_atm - G_beam
```

### 3.2 频率复用技术 (Frequency Reuse)

#### 3.2.1 多色频率复用

传统卫星系统采用4色或7色频率复用方案降低同频干扰：

```
┌─────────────────────────────────────────┐
│           7色频率复用示意图              │
│                                          │
│         f1    f2    f3                   │
│           ╲   │   ╱                      │
│        f7  ╲  │  ╱  f4                   │
│           ╲ ╲│╱ ╱                        │
│        ────╳═╳═╳────  中心波束(f0)       │
│           ╱ ╱│╲ ╲                        │
│        f6  ╱  │  ╲  f5                   │
│           ╱   │   ╲                      │
│                                          │
│  频谱效率：SE = 1/N (N为复用因子)        │
└─────────────────────────────────────────┘
```

#### 3.2.2 全频复用+预编码

6G趋势：采用全频复用(N=1)配合多波束预编码消除干扰

**干扰模型对应原题**：
```
原题干扰模型：
γ = (p_n,k × φ_n,k × |h_n,k|²) / (Σ_{u≠n} p_u,k × φ_u,k × |h_u,k|² + N_0)

6G星地干扰模型：
γ_sat = (P_tx × G_tx × G_rx × |h|²) / (Σ_interference + N_0)

其中干扰项包括：
1. 波束间干扰(IBI): 同卫星不同波束
2. 星间干扰(ISI): 不同卫星同频波束  ← 对应原题"微基站间干扰"
3. 星地干扰(LTI): 与地面网络共频
```

### 3.3 高波束增益技术

#### 3.3.1 相控阵天线与波束赋形

```
大规模MIMO波束赋形：

发射信号: x = W × s
其中:
- W ∈ C^{N_t × K}: 预编码矩阵
- s ∈ C^{K × 1}: K个用户的信息符号
- N_t: 发射天线数 (典型值: 256-1024)

波束增益: G = 10×log10(N_t) + G_element

示例:
- 256元相控阵: G ≈ 24 dB + 5 dB = 29 dB (元件增益)
- 1024元相控阵: G ≈ 30 dB + 5 dB = 35 dB
```

#### 3.3.2 与原题资源块的对应

原题的"资源块(RB)"可以重新定义为**波束-时间-频率**三维资源单元：

```
6G资源块定义：
┌──────────────────────────────────────────┐
│   Resource Unit (RU) = (Beam, Time, Freq) │
│                                           │
│   频域: Δf = 360 kHz (子载波带宽)         │
│   时域: T_slot = 10 ms (时隙)            │
│   空域: Beam_id ∈ {1, 2, ..., N_beam}    │
│                                           │
│   原题50个RB → 50个RU可分配给不同波束     │
└──────────────────────────────────────────┘
```

### 3.4 高阶调制与自适应编码调制(ACM)

#### 3.4.1 调制方案

| 调制方式 | 频谱效率 | 所需SNR (BER=10⁻⁶) | 适用场景 |
|---------|---------|-------------------|---------|
| QPSK | 2 bit/s/Hz | 10.5 dB | 边缘用户/恶劣信道 |
| 8PSK | 3 bit/s/Hz | 14.0 dB | 一般信道 |
| 16APSK | 4 bit/s/Hz | 16.5 dB | 良好信道 |
| 32APSK | 5 bit/s/Hz | 19.5 dB | 优质信道 |
| 64APSK | 6 bit/s/Hz | 22.5 dB | 6G目标 |
| 256QAM | 8 bit/s/Hz | 28.0 dB | 6G高阶目标 |

#### 3.4.2 ACM与原题速率计算的对应

原题香农公式：
```
r = i×b × log₂(1 + γ)
```

6G ACM实际速率：
```
r_ACM = η(MODCOD) × B × (1 - FER)

其中:
- η(MODCOD): 编码调制效率查表
- B: 带宽
- FER: 帧错误率

DVB-S2X标准定义了28种MODCOD组合，
效率范围: 0.43 - 5.51 bit/s/Hz
```

---

## 四、空地信道编码技术

### 4.1 6G信道编码方案对比

| 编码类型 | 5G用途 | 6G空天适用性 | 优势 | 劣势 |
|---------|-------|-------------|------|------|
| **LDPC** | 数据信道 | ★★★★★ | 高吞吐、低误码平层 | 短码性能差 |
| **Polar** | 控制信道 | ★★★★☆ | 理论最优、可达容量 | 串行译码延迟 |
| **Turbo** | 4G遗留 | ★★★☆☆ | 成熟稳定 | 误码平层 |
| **GLDPC-PC** | 6G研究 | ★★★★★ | 结合LDPC+Polar优势 | 尚未标准化 |

### 4.2 星地链路编码特殊考虑

```
星地信道特点：
1. 长传播延迟 (LEO: 3-10ms, GEO: 250ms)
   → ARQ不适用，必须强FEC
   
2. 大多普勒频移 (LEO: ±40kHz)
   → OTFS调制 + 交织编码

3. 大气湍流/雨衰
   → 深度交织 + ACM

推荐编码方案：
- URLLC切片: Polar码 (K=64-256), SCL译码
- eMBB切片: LDPC码 (K=8192), 码率0.5-0.9
- mMTC切片: Polar码 + Grant-Free NOMA
```

### 4.3 编码参数与原题QoS的对应

原题QoS定义中的可靠性要求可映射为编码参数：

```python
# URLLC切片: 高可靠低时延
urllc_coding = {
    'code_type': 'Polar',
    'code_length': 256,
    'code_rate': 0.5,
    'target_BLER': 1e-5,  # 对应原题α^t惩罚
    'max_iterations': 8,
    'decoder': 'SCL-8'
}

# eMBB切片: 高吞吐
embb_coding = {
    'code_type': 'LDPC',
    'code_length': 8192,
    'code_rate': 0.75,
    'target_BLER': 1e-3,
    'max_iterations': 50,
    'decoder': 'Min-Sum'
}

# mMTC切片: 海量连接
mmtc_coding = {
    'code_type': 'Polar',
    'code_length': 128,
    'code_rate': 0.33,
    'target_BLER': 1e-2,
    'max_iterations': 4,
    'decoder': 'SC'  # 简化译码降低复杂度
}
```

---

## 五、网络切片资源管理模型重构

### 5.1 三类切片在空天网络中的应用

| 切片类型 | 6G空天应用场景 | 关键指标 | 资源需求特征 |
|---------|--------------|---------|-------------|
| **URLLC** | 远程手术、自动驾驶、工业控制 | 时延<1ms, 可靠性99.9999% | 预留资源、优先调度 |
| **eMBB** | 卫星宽带、AR/VR、8K视频 | 速率>1Gbps, 时延<10ms | 大带宽、高效复用 |
| **mMTC** | 卫星物联网、全球资产追踪 | 连接密度10⁶/km², 功耗极低 | Grant-Free接入 |

### 5.2 优化目标函数重构

**原题目标**：最大化用户服务质量

**6G空天重构目标**：

```
max  Σ_k ω_k × Q_k(r_k, d_k)  - λ × E_total

s.t.  
      Σ_b N_b^slice ≤ N_total        (资源约束)
      p_b ∈ [P_min, P_max]           (功率约束)  
      d_k^URLLC ≤ 1 ms               (URLLC时延)
      r_k^eMBB ≥ 50 Mbps             (eMBB速率)
      Σ_k a_k^mMTC / Σ_k b_k ≥ 0.99  (mMTC接入率)
      I_inter-beam ≤ I_threshold      (波束间干扰)

其中:
- Q_k: 用户k的服务质量函数 (沿用原题定义)
- E_total: 系统总能耗 (6G绿色通信约束)
- λ: 能效权重因子
```

### 5.3 决策变量重构

| 原题决策变量 | 6G空天决策变量 | 说明 |
|-------------|--------------|------|
| 切片RB分配 | 波束-频率-时间资源分配 | 三维资源管理 |
| 基站功率 | 波束功率+预编码矩阵 | 干扰协调 |
| 用户接入(MBS/SBS) | 用户接入(卫星/地面/HAPS) | 异构接入选择 |

---

## 六、数据集改造方案

### 6.1 信道数据改造

**原数据**：`channel_data.xlsx`, `BS1/2/3.xlsx`, `MBS_1.xlsx`, `SBS_1/2/3.xlsx`

**改造方案**：

```python
def transform_channel_data(original_data):
    """
    将原5G信道数据改造为6G星地信道数据
    """
    transformed = {}
    
    # 1. 大规模衰减 → 星地路径损耗
    # 原: φ_n,k (dB), 范围约30-80dB
    # 新: L_sat (dB), 范围约160-190dB
    transformed['path_loss'] = original_data['大规模衰减'] + 130  # 偏移量
    
    # 2. 小规模瑞利衰减 → Shadowed-Rician衰落
    # 原: h_n,k, 瑞利分布
    # 新: 莱斯因子K=10dB的Shadowed-Rician
    # 通过添加确定性分量模拟LOS主径
    h_original = original_data['小规模瑞丽衰减']
    K_factor = 10  # 莱斯因子 (dB)
    K_linear = 10**(K_factor/10)
    h_los = np.sqrt(K_linear / (1 + K_linear))  # LOS分量
    h_nlos = np.sqrt(1 / (1 + K_linear)) * h_original  # NLOS分量
    transformed['fading'] = h_los + h_nlos
    
    # 3. 用户位置 → 波束覆盖区内位置 + 多普勒
    # 添加用户相对运动速度分量 (模拟LEO高速运动)
    v_sat = 7.5e3  # m/s, LEO轨道速度
    f_c = 30e9  # Hz, Ka波段载频
    c = 3e8
    elevation = np.arctan(500e3 / original_data['距离'])  # 仰角
    doppler_shift = (v_sat / c) * f_c * np.cos(elevation)
    transformed['doppler'] = doppler_shift
    
    return transformed
```

### 6.2 任务流数据改造

**原数据**：`taskflow.xlsx` - 用户任务到达

**6G场景增强**：

```python
def transform_task_flow(original_taskflow):
    """
    增强任务流数据以反映6G场景特征
    """
    enhanced = original_taskflow.copy()
    
    # 1. URLLC任务：添加紧急等级
    enhanced['URLLC_priority'] = np.random.choice([1,2,3], 
                                   size=len(enhanced['U1']),
                                   p=[0.7, 0.2, 0.1])
    
    # 2. eMBB任务：添加视频编码类型
    enhanced['eMBB_codec'] = np.random.choice(['H.265', 'H.266', 'AV1'],
                                size=len(enhanced['e1']))
    
    # 3. mMTC任务：添加设备类型和duty cycle
    enhanced['mMTC_device_type'] = np.random.choice(
        ['sensor', 'tracker', 'meter'],
        size=len(enhanced['m1'])
    )
    enhanced['mMTC_duty_cycle'] = np.random.uniform(0.001, 0.1, 
                                   size=len(enhanced['m1']))
    
    return enhanced
```

---

## 七、关键技术参考文献

### 7.1 顶会/顶刊论文（2023-2025）

1. **IEEE JSAC 2024**: "Spectrum Sharing and Interference Management for 6G LEO Satellite-Terrestrial Network Integration"
   - 干扰类型：IBI, ISI, LTI
   - 频谱共享策略

2. **IEEE TWC 2024**: "Dynamic Interference Prediction and Receive Beamforming for Dense LEO Satellite Networks"
   - LSTM干扰预测
   - 深度强化学习波束赋形

3. **Science China Information Sciences 2023**: "Coverage enhancement for 6G satellite-terrestrial integrated networks"
   - 覆盖性能指标
   - 星座配置优化

4. **IEEE Wireless Communications 2023**: "Satellite-Terrestrial Integrated 6G: An Ultra-Dense LEO Networking Management Architecture"
   - MEO-LEO-SES分层管理
   - 资源管理框架

5. **arXiv 2024**: "Channel Coding Toward 6G: Technical Overview and Outlook"
   - LDPC/Polar码对比
   - 6G编码需求分析

6. **IEEE Access 2024**: "Advanced Channel Coding Schemes for B5G/6G Networks"
   - 编码技术演进
   - 星地链路适用性

### 7.2 技术标准

1. **3GPP TR 38.811**: Study on Non-Terrestrial Networks (NTN)
2. **DVB-S2X**: Digital Video Broadcasting - Satellite Second Generation Extension
3. **ITU-R M.2083**: IMT Vision – Framework for 6G

## 附录：核心公式汇总

### A.1 星地链路预算

```
P_rx = P_tx + G_tx + G_rx - L_fs - L_atm - L_rain - L_pointing

其中:
- P_tx: 发射功率 (dBW)
- G_tx: 发射天线增益 (dBi)  
- G_rx: 接收天线增益 (dBi)
- L_fs: 自由空间损耗 = 20log(4πd/λ) (dB)
- L_atm: 大气吸收损耗 (dB)
- L_rain: 雨衰 (dB)
- L_pointing: 指向损耗 (dB)
```

### A.2 系统容量

```
C = Σ_b Σ_k W × log₂(1 + SINR_k,b)

SINR_k,b = (P_b × G_b(θ_k) × |h_k|²) / (I_IBI + I_ISI + N_0)
```

### A.3 QoS函数 (沿用原题)

```
URLLC: Q = α^(d-d_queue), d ≤ d_SLA; -M_URLLC, d > d_SLA
eMBB:  Q = 1, r ≥ r_SLA; r/r_SLA, r < r_SLA; -M_eMBB, d > d_SLA  
mMTC:  Q = Σa'_i / Σa_i, d ≤ d_SLA; -M_mMTC, d > d_SLA
```
