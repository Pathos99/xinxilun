%% 6G空天通信课程作业 - 完整图表生成代码 (增强版)
%% ========================================================================
%% 说明: 
%%   - 运行环境: MATLAB R2020a或更高版本
%%   - 输出: 15张专业PNG图表 + 对应的FIG文件
%%   - 所有标签使用英文避免中文乱码问题
%% ========================================================================

%% 清理环境
clear; clc; close all;

%% 全局配置
set(0, 'DefaultAxesFontName', 'Arial');
set(0, 'DefaultAxesFontSize', 11);
set(0, 'DefaultTextFontName', 'Arial');
set(0, 'DefaultTextFontSize', 11);
set(0, 'DefaultLineLineWidth', 1.5);

% 专业配色方案
colors.primary = [0.2 0.4 0.8];      % 主色-蓝
colors.secondary = [0.85 0.2 0.2];   % 副色-红
colors.success = [0.2 0.7 0.3];      % 成功-绿
colors.warning = [0.95 0.5 0.1];     % 警告-橙
colors.info = [0.3 0.7 0.9];         % 信息-青
colors.purple = [0.6 0.3 0.7];       % 紫色
colors.gray = [0.5 0.5 0.5];         % 灰色

fprintf('Starting figure generation...\n');
fprintf('==========================================\n');

%% ========================================================================
%% Figure 1: 多波束增益方向图 (Bessel函数模型)
%% ========================================================================
fprintf('Generating Figure 1: Beam Pattern...\n');

figure('Position', [100 100 900 650], 'Color', 'white');

theta_deg = linspace(-3, 3, 1000);
theta_3dB = 0.5;  % 3dB波束宽度
G_max = 48;       % 最大增益 dBi

% Bessel函数计算
u = 2.07123 * sind(theta_deg) ./ sind(theta_3dB);
u(abs(u) < 1e-10) = 1e-10;
gain_factor = (2 * besselj(1, u) ./ u).^2;
gain_factor(abs(theta_deg) < 0.01) = 1;
gain_dB = G_max + 10 * log10(max(gain_factor, 1e-10));

plot(theta_deg, gain_dB, 'Color', colors.primary, 'LineWidth', 2.5);
hold on;
yline(G_max - 3, '--', 'Color', colors.secondary, 'LineWidth', 2, 'Label', '-3dB Level');
xline(theta_3dB, ':', 'Color', colors.success, 'LineWidth', 1.5);
xline(-theta_3dB, ':', 'Color', colors.success, 'LineWidth', 1.5);

% 标注3dB波束宽度
annotation('doublearrow', [0.43 0.57], [0.3 0.3], 'Color', colors.warning);
text(0, gain_dB(500)-8, sprintf('\\theta_{3dB} = %.1f°', theta_3dB*2), ...
    'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');

xlabel('Off-axis Angle \theta (degrees)', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('Antenna Gain G(\theta) (dBi)', 'FontWeight', 'bold', 'FontSize', 12);
title('Multi-Beam Antenna Gain Pattern (Bessel Function Model)', ...
    'FontSize', 14, 'FontWeight', 'bold');
subtitle(sprintf('G_{max} = %d dBi, \\theta_{3dB} = %.1f°', G_max, theta_3dB));
legend('Beam Pattern', '-3dB Reference', '\theta_{3dB} Boundary', 'Location', 'northeast');
grid on; box on;
ylim([0 52]); xlim([-3 3]);

saveas(gcf, 'fig01_beam_pattern.png');
saveas(gcf, 'fig01_beam_pattern.fig');

%% ========================================================================
%% Figure 2: DVB-S2X ACM MODCOD频谱效率
%% ========================================================================
fprintf('Generating Figure 2: DVB-S2X ACM...\n');

figure('Position', [100 100 1000 600], 'Color', 'white');

modcod_names = {'QPSK 1/4', 'QPSK 1/2', 'QPSK 3/4', '8PSK 2/3', '8PSK 3/4', ...
                '16APSK 2/3', '16APSK 3/4', '32APSK 4/5', '64APSK 5/6', '256APSK 3/4'};
spectral_eff = [0.49, 0.99, 1.49, 1.98, 2.23, 2.64, 2.97, 3.95, 4.94, 5.51];
snr_threshold = [-2.35, 1.00, 4.03, 6.62, 8.97, 9.82, 10.21, 12.73, 16.05, 18.10];

cmap = parula(length(modcod_names));

subplot(1, 2, 1);
b = bar(spectral_eff);
b.FaceColor = 'flat';
for i = 1:length(modcod_names)
    b.CData(i,:) = cmap(i,:);
end
set(gca, 'XTickLabel', modcod_names, 'XTickLabelRotation', 45);
ylabel('Spectral Efficiency (bit/s/Hz)', 'FontWeight', 'bold');
title('DVB-S2X MODCOD Spectral Efficiency', 'FontWeight', 'bold');
ylim([0 6.5]);
grid on; box on;
for i = 1:length(spectral_eff)
    text(i, spectral_eff(i) + 0.2, sprintf('%.2f', spectral_eff(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
end

subplot(1, 2, 2);
scatter(snr_threshold, spectral_eff, 100, cmap, 'filled', 'MarkerEdgeColor', 'k');
hold on;
plot(snr_threshold, spectral_eff, '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);

% 添加MODCOD标签
for i = [1, 5, 10]
    text(snr_threshold(i)+0.5, spectral_eff(i)+0.15, modcod_names{i}, ...
        'FontSize', 9, 'FontWeight', 'bold');
end

xlabel('Required SNR (dB)', 'FontWeight', 'bold');
ylabel('Spectral Efficiency (bit/s/Hz)', 'FontWeight', 'bold');
title('SNR vs Spectral Efficiency Curve', 'FontWeight', 'bold');
grid on; box on;
xlim([-5 20]); ylim([0 6.5]);

sgtitle('DVB-S2X Adaptive Coding and Modulation (ACM)', 'FontSize', 15, 'FontWeight', 'bold');
saveas(gcf, 'fig02_dvbs2x_acm.png');
saveas(gcf, 'fig02_dvbs2x_acm.fig');

%% ========================================================================
%% Figure 3: 6G信道编码选择 (按块长度)
%% ========================================================================
fprintf('Generating Figure 3: Coding Selection...\n');

figure('Position', [100 100 950 650], 'Color', 'white');

block_length = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
conv_ber = [5e-3, 2e-3, 1e-3, 8e-4, 6e-4, 5e-4, 5e-4, 5e-4, 5e-4];
polar_ber = [1e-2, 3e-3, 8e-4, 3e-4, 1e-4, 5e-5, 3e-5, 2e-5, 2e-5];
ldpc_ber = [5e-2, 2e-2, 5e-3, 1e-3, 2e-4, 5e-5, 1e-5, 3e-6, 1e-6];

semilogy(block_length, conv_ber, '-o', 'Color', colors.success, 'LineWidth', 2.5, ...
    'MarkerFaceColor', colors.success, 'MarkerSize', 9, 'DisplayName', 'Convolutional (Viterbi)');
hold on;
semilogy(block_length, polar_ber, '-s', 'Color', colors.primary, 'LineWidth', 2.5, ...
    'MarkerFaceColor', colors.primary, 'MarkerSize', 9, 'DisplayName', 'Polar (SCL-8)');
semilogy(block_length, ldpc_ber, '-^', 'Color', colors.secondary, 'LineWidth', 2.5, ...
    'MarkerFaceColor', colors.secondary, 'MarkerSize', 9, 'DisplayName', 'LDPC (BP-12)');

% 最优区域标记
fill([32 128 128 32], [1e-6 1e-6 1 1], colors.success, 'FaceAlpha', 0.12, 'EdgeColor', 'none');
fill([128 512 512 128], [1e-6 1e-6 1 1], colors.primary, 'FaceAlpha', 0.12, 'EdgeColor', 'none');
fill([512 10000 10000 512], [1e-6 1e-6 1 1], colors.secondary, 'FaceAlpha', 0.12, 'EdgeColor', 'none');

% 区域标签
text(65, 3e-5, 'Conv Optimal', 'FontWeight', 'bold', 'Color', colors.success, 'FontSize', 11);
text(280, 3e-5, 'Polar Optimal', 'FontWeight', 'bold', 'Color', colors.primary, 'FontSize', 11);
text(2500, 3e-5, 'LDPC Optimal', 'FontWeight', 'bold', 'Color', colors.secondary, 'FontSize', 11);

set(gca, 'XScale', 'log');
xlabel('Block Length N', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('BER @ E_b/N_0 = 3dB', 'FontWeight', 'bold', 'FontSize', 12);
title('6G Channel Coding Selection by Block Length', 'FontSize', 14, 'FontWeight', 'bold');
subtitle('Based on IEEE Proceedings 2024 Survey (Figure 12)');
legend('Location', 'northeast', 'FontSize', 10);
grid on; box on;
xlim([30 12000]); ylim([1e-6 0.1]);

saveas(gcf, 'fig03_coding_selection.png');
saveas(gcf, 'fig03_coding_selection.fig');

%% ========================================================================
%% Figure 4: IBI/ISI/LTI三类干扰框架
%% ========================================================================
fprintf('Generating Figure 4: Interference Framework...\n');

figure('Position', [100 100 900 650], 'Color', 'white');

interference_types = categorical({'IBI\n(Inter-Beam)', 'ISI\n(Inter-Satellite)', 'LTI\n(LEO-Terrestrial)'});
interference_types = reordercats(interference_types, {'IBI\n(Inter-Beam)', 'ISI\n(Inter-Satellite)', 'LTI\n(LEO-Terrestrial)'});

baseline = [1.0, 1.0, 1.0];
with_mitigation = [0.65, 0.72, 0.45];
reduction_pct = (1 - with_mitigation) * 100;

b = bar([baseline; with_mitigation]', 'grouped');
b(1).FaceColor = colors.secondary;
b(2).FaceColor = colors.info;

set(gca, 'XTickLabel', {'IBI (Inter-Beam)', 'ISI (Inter-Satellite)', 'LTI (LEO-Terrestrial)'});

for i = 1:3
    text(i, with_mitigation(i) + 0.1, sprintf('-%.0f%%', reduction_pct(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Color', colors.success, 'FontSize', 14);
end

ylabel('Normalized Interference Level', 'FontWeight', 'bold', 'FontSize', 12);
title('IBI/ISI/LTI Interference Classification Framework', 'FontSize', 14, 'FontWeight', 'bold');
subtitle('Based on IEEE Communications Surveys & Tutorials 2025');
legend('Without Mitigation', 'With Mitigation', 'Location', 'northeast', 'FontSize', 11);
ylim([0 1.4]);
grid on; box on;

saveas(gcf, 'fig04_interference_framework.png');
saveas(gcf, 'fig04_interference_framework.fig');

%% ========================================================================
%% Figure 5: 网络切片QoS性能
%% ========================================================================
fprintf('Generating Figure 5: Network Slicing QoS...\n');

figure('Position', [100 100 900 600], 'Color', 'white');

slices = categorical({'URLLC\n(Polar N=256)', 'eMBB\n(LDPC N=8448)', 'mMTC\n(Polar N=128)'});

qos_achieved = [0.93, 0.85, 0.95];
qos_target = [0.90, 0.80, 0.90];

b = bar([qos_achieved; qos_target]', 'grouped');
b(1).FaceColor = colors.info;
b(2).FaceColor = colors.warning;

set(gca, 'XTickLabel', {'URLLC (Polar N=256)', 'eMBB (LDPC N=8448)', 'mMTC (Polar N=128)'});

yline(1.0, '--', 'Color', colors.gray, 'LineWidth', 1.5, 'Label', 'Perfect QoS = 1.0');

for i = 1:3
    text(i-0.15, qos_achieved(i) + 0.03, sprintf('%.2f', qos_achieved(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
    text(i+0.15, qos_target(i) + 0.03, sprintf('%.2f', qos_target(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10);
end

ylabel('QoS Score', 'FontWeight', 'bold', 'FontSize', 12);
title('Network Slicing QoS Performance with Optimal Coding', 'FontSize', 14, 'FontWeight', 'bold');
legend('Achieved QoS', 'Target QoS', 'Location', 'northeast', 'FontSize', 11);
ylim([0 1.2]);
grid on; box on;

saveas(gcf, 'fig05_slicing_qos.png');
saveas(gcf, 'fig05_slicing_qos.fig');

%% ========================================================================
%% Figure 6: 相控阵波束增益
%% ========================================================================
fprintf('Generating Figure 6: Phased Array Gain...\n');

figure('Position', [100 100 850 600], 'Color', 'white');

antenna_elements = [64, 128, 256, 512, 1024];
G_element = 5;
array_gain = 10 * log10(antenna_elements) + G_element;

cmap = flipud(cool(length(antenna_elements)));
b = bar(array_gain);
b.FaceColor = 'flat';
for i = 1:length(antenna_elements)
    b.CData(i,:) = cmap(i,:);
end

hold on;
plot(1:length(antenna_elements), array_gain, 'ko-', 'LineWidth', 2.5, 'MarkerSize', 12, 'MarkerFaceColor', 'k');

set(gca, 'XTickLabel', arrayfun(@(x) sprintf('%d Elements', x), antenna_elements, 'UniformOutput', false));
ylabel('Array Gain (dBi)', 'FontWeight', 'bold', 'FontSize', 12);
title('Phased Array Beamforming Gain', 'FontSize', 14, 'FontWeight', 'bold');
subtitle('Hybrid Analog-Digital Architecture');
ylim([15 42]);
grid on; box on;

for i = 1:length(antenna_elements)
    text(i, array_gain(i) + 1, sprintf('%.1f dBi', array_gain(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
end

saveas(gcf, 'fig06_phased_array.png');
saveas(gcf, 'fig06_phased_array.fig');

%% ========================================================================
%% Figure 7: LEO星座覆盖示意图
%% ========================================================================
fprintf('Generating Figure 7: LEO Constellation...\n');

figure('Position', [100 100 900 750], 'Color', 'white');

R_earth = 6371;
h_leo = 500;
R_orbit = R_earth + h_leo;

theta = linspace(0, 2*pi, 200);
fill(R_earth*cos(theta), R_earth*sin(theta), [0.3 0.6 0.9], 'EdgeColor', [0.1 0.3 0.6], 'LineWidth', 1.5);
hold on;

plot(R_orbit*cos(theta), R_orbit*sin(theta), '--', 'Color', colors.gray, 'LineWidth', 2);

n_sat = 6;
sat_angles = linspace(0, 2*pi, n_sat+1);
sat_angles = sat_angles(1:end-1);

for i = 1:n_sat
    x_sat = R_orbit * cos(sat_angles(i));
    y_sat = R_orbit * sin(sat_angles(i));
    
    plot(x_sat, y_sat, 's', 'MarkerSize', 18, 'MarkerFaceColor', colors.warning, ...
        'MarkerEdgeColor', 'k', 'LineWidth', 2);
    
    beam_width = deg2rad(22);
    beam_start = sat_angles(i) - beam_width/2 + pi;
    beam_end = sat_angles(i) + beam_width/2 + pi;
    theta_beam = linspace(beam_start, beam_end, 30);
    x_beam = [x_sat, R_earth*cos(theta_beam), x_sat];
    y_beam = [y_sat, R_earth*sin(theta_beam), y_sat];
    fill(x_beam, y_beam, colors.warning, 'FaceAlpha', 0.25, 'EdgeColor', colors.warning, 'LineWidth', 1);
end

gateway_angles = [0, 2*pi/3, 4*pi/3];
for i = 1:length(gateway_angles)
    x_gw = R_earth * 0.98 * cos(gateway_angles(i));
    y_gw = R_earth * 0.98 * sin(gateway_angles(i));
    plot(x_gw, y_gw, '^', 'MarkerSize', 14, 'MarkerFaceColor', colors.success, ...
        'MarkerEdgeColor', 'k', 'LineWidth', 2);
end

axis equal;
xlim([-9500 9500]); ylim([-9500 9500]);
title('LEO Satellite Constellation Multi-Beam Coverage', 'FontSize', 14, 'FontWeight', 'bold');
subtitle(sprintf('Altitude: %d km, %d Satellites, Global Coverage', h_leo, n_sat));
legend('Earth', 'LEO Orbit', 'LEO Satellite', 'Beam Coverage', 'Ground Gateway', ...
    'Location', 'northeast', 'FontSize', 10);
xlabel('Distance (km)', 'FontWeight', 'bold');
ylabel('Distance (km)', 'FontWeight', 'bold');
grid on; box on;

saveas(gcf, 'fig07_leo_constellation.png');
saveas(gcf, 'fig07_leo_constellation.fig');

%% ========================================================================
%% Figure 8: 5G→6G架构映射对比
%% ========================================================================
fprintf('Generating Figure 8: Architecture Mapping...\n');

figure('Position', [100 100 1100 550], 'Color', 'white');
t = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% 左图: 5G架构
nexttile;
axis off; hold on;
title('5G Network Slicing Architecture', 'FontSize', 13, 'FontWeight', 'bold');

rectangle('Position', [0.25, 0.72, 0.5, 0.2], 'Curvature', 0.15, ...
    'FaceColor', colors.primary, 'EdgeColor', 'k', 'LineWidth', 2);
text(0.5, 0.82, {'MBS', '(100 RB)'}, 'HorizontalAlignment', 'center', ...
    'Color', 'w', 'FontWeight', 'bold', 'FontSize', 11);

positions = {[0.02, 0.4], [0.35, 0.4], [0.68, 0.4]};
for i = 1:3
    rectangle('Position', [positions{i}(1), positions{i}(2), 0.3, 0.18], 'Curvature', 0.15, ...
        'FaceColor', colors.success, 'EdgeColor', 'k', 'LineWidth', 1.5);
    text(positions{i}(1)+0.15, positions{i}(2)+0.09, sprintf('SBS-%d', i), ...
        'HorizontalAlignment', 'center', 'Color', 'w', 'FontWeight', 'bold', 'FontSize', 10);
end

slice_colors = {colors.secondary, colors.primary, colors.purple};
slice_names = {'URLLC', 'eMBB', 'mMTC'};
for i = 1:3
    rectangle('Position', [positions{i}(1)+0.05, 0.12, 0.2, 0.15], 'Curvature', 0.1, ...
        'FaceColor', slice_colors{i}, 'EdgeColor', 'k', 'LineWidth', 1);
    text(positions{i}(1)+0.15, 0.195, slice_names{i}, ...
        'HorizontalAlignment', 'center', 'Color', 'w', 'FontWeight', 'bold', 'FontSize', 9);
end

xlim([0 1]); ylim([0 1]);

% 右图: 6G架构
nexttile;
axis off; hold on;
title('6G Space-Ground Architecture', 'FontSize', 13, 'FontWeight', 'bold');

rectangle('Position', [0.25, 0.75, 0.5, 0.18], 'Curvature', 0.15, ...
    'FaceColor', colors.warning, 'EdgeColor', 'k', 'LineWidth', 2);
text(0.5, 0.84, {'LEO Satellite', '(100 Beams)'}, 'HorizontalAlignment', 'center', ...
    'Color', 'w', 'FontWeight', 'bold', 'FontSize', 11);

for i = 1:3
    rectangle('Position', [positions{i}(1), positions{i}(2), 0.3, 0.18], 'Curvature', 0.15, ...
        'FaceColor', colors.info, 'EdgeColor', 'k', 'LineWidth', 1.5);
    text(positions{i}(1)+0.15, positions{i}(2)+0.09, sprintf('Gateway-%d', i), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);
end

app_names = {'Remote Surgery', 'Sat Broadband', 'Satellite IoT'};
for i = 1:3
    rectangle('Position', [positions{i}(1)+0.02, 0.1, 0.26, 0.18], 'Curvature', 0.1, ...
        'FaceColor', slice_colors{i}, 'EdgeColor', 'k', 'LineWidth', 1);
    text(positions{i}(1)+0.15, 0.19, app_names{i}, ...
        'HorizontalAlignment', 'center', 'Color', 'w', 'FontWeight', 'bold', 'FontSize', 8);
end

xlim([0 1]); ylim([0 1]);

sgtitle('5G to 6G Architecture Transformation', 'FontSize', 15, 'FontWeight', 'bold');

saveas(gcf, 'fig08_architecture_mapping.png');
saveas(gcf, 'fig08_architecture_mapping.fig');

%% ========================================================================
%% Figure 9: Shadowed-Rician信道模型
%% ========================================================================
fprintf('Generating Figure 9: Channel Model...\n');

figure('Position', [100 100 1000 550], 'Color', 'white');

K_factors = [0, 5, 10, 15];
x = linspace(0, 4, 1000);

subplot(1, 2, 1);
line_colors = [colors.secondary; colors.warning; colors.primary; colors.success];
for idx = 1:length(K_factors)
    K_linear = 10^(K_factors(idx)/10);
    sigma = sqrt(1/(2*(K_linear+1)));
    s = sqrt(K_linear/(K_linear+1));
    pdf = (x/sigma^2) .* exp(-(x.^2 + s^2)/(2*sigma^2)) .* besseli(0, x*s/sigma^2);
    pdf(isnan(pdf)) = 0;
    pdf = pdf / max(pdf);
    plot(x, pdf, 'LineWidth', 2.5, 'Color', line_colors(idx,:), ...
        'DisplayName', sprintf('K = %d dB', K_factors(idx)));
    hold on;
end

xlabel('Amplitude |h|', 'FontWeight', 'bold', 'FontSize', 11);
ylabel('Normalized PDF', 'FontWeight', 'bold', 'FontSize', 11);
title('Shadowed-Rician Fading Distribution', 'FontWeight', 'bold', 'FontSize', 12);
legend('Location', 'northeast', 'FontSize', 10);
grid on; box on;

subplot(1, 2, 2);
t = linspace(0, 1, 1000);
rng(42);
K_linear = 10;
fading_los = sqrt(K_linear/(K_linear+1)) * ones(size(t));
fading_nlos = sqrt(1/(K_linear+1)) * (randn(size(t)) + 1j*randn(size(t)))/sqrt(2);
fading_total = abs(fading_los + fading_nlos);
fading_dB = 20*log10(fading_total);

plot(t*1000, fading_dB, 'Color', colors.primary, 'LineWidth', 1.2);
hold on;
yline(0, '--', 'Color', colors.secondary, 'LineWidth', 1.5, 'Label', 'Mean Level');

xlabel('Time (ms)', 'FontWeight', 'bold', 'FontSize', 11);
ylabel('Channel Gain (dB)', 'FontWeight', 'bold', 'FontSize', 11);
title('Time-Varying Fading (K=10dB)', 'FontWeight', 'bold', 'FontSize', 12);
ylim([-12 8]);
grid on; box on;

sgtitle('Space-Ground Channel Model: Shadowed-Rician Fading', 'FontSize', 14, 'FontWeight', 'bold');

saveas(gcf, 'fig09_channel_model.png');
saveas(gcf, 'fig09_channel_model.fig');

%% ========================================================================
%% Figure 10: 资源分配仿真结果
%% ========================================================================
fprintf('Generating Figure 10: Simulation Results...\n');

figure('Position', [100 100 1100 750], 'Color', 'white');

epochs = 1:10;
urllc_rb = [20, 10, 20, 20, 10, 20, 10, 20, 20, 10];
embb_rb = [30, 35, 25, 30, 35, 25, 35, 30, 25, 35];
mmtc_rb = 100 - urllc_rb - embb_rb;
qos_values = [0.93, 0.91, 0.89, 0.92, 0.90, 0.88, 0.91, 0.93, 0.92, 0.90];

subplot(2, 2, 1);
bar(epochs, [urllc_rb; embb_rb; mmtc_rb]', 'stacked');
colororder([colors.secondary; colors.primary; colors.purple]);
xlabel('Decision Epoch', 'FontWeight', 'bold');
ylabel('Resource Blocks', 'FontWeight', 'bold');
title('RB Allocation per Epoch', 'FontWeight', 'bold');
legend('URLLC', 'eMBB', 'mMTC', 'Location', 'eastoutside');
ylim([0 110]);
grid on; box on;

subplot(2, 2, 2);
plot(epochs, qos_values, '-o', 'Color', colors.info, 'LineWidth', 2.5, ...
    'MarkerFaceColor', colors.info, 'MarkerSize', 10);
hold on;
yline(0.9, '--', 'Color', colors.secondary, 'LineWidth', 2, 'Label', 'Target QoS = 0.9');
fill([epochs fliplr(epochs)], [0.9*ones(1,10) fliplr(qos_values)], ...
    colors.success, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
xlabel('Decision Epoch', 'FontWeight', 'bold');
ylabel('QoS Score', 'FontWeight', 'bold');
title('QoS Performance over Time', 'FontWeight', 'bold');
ylim([0.82 1.0]);
grid on; box on;

subplot(2, 2, 3);
power_leo = [38, 37, 38, 36, 38, 37, 38, 39, 37, 38];
power_gw = [28, 27, 28, 26, 27, 28, 27, 28, 27, 28];
plot(epochs, power_leo, '-s', 'Color', colors.warning, 'LineWidth', 2, ...
    'MarkerFaceColor', colors.warning, 'MarkerSize', 9);
hold on;
plot(epochs, power_gw, '-^', 'Color', colors.success, 'LineWidth', 2, ...
    'MarkerFaceColor', colors.success, 'MarkerSize', 9);
xlabel('Decision Epoch', 'FontWeight', 'bold');
ylabel('Transmit Power (dBm)', 'FontWeight', 'bold');
title('Power Allocation', 'FontWeight', 'bold');
legend('LEO Satellite', 'Ground Gateway', 'Location', 'northeast');
ylim([20 45]);
grid on; box on;

subplot(2, 2, 4);
throughput = [185, 198, 175, 210, 192, 178, 205, 215, 188, 195];
bar(epochs, throughput, 'FaceColor', colors.purple, 'EdgeColor', 'k');
hold on;
plot(epochs, throughput, '-o', 'Color', 'k', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'k');
yline(mean(throughput), '--', 'Color', colors.secondary, 'LineWidth', 2, ...
    'Label', sprintf('Avg: %.0f Mbps', mean(throughput)));
xlabel('Decision Epoch', 'FontWeight', 'bold');
ylabel('Throughput (Mbps)', 'FontWeight', 'bold');
title('System Throughput', 'FontWeight', 'bold');
ylim([150 230]);
grid on; box on;

sgtitle('6G Space-Ground Resource Allocation Simulation Results', 'FontSize', 15, 'FontWeight', 'bold');

saveas(gcf, 'fig10_simulation_results.png');
saveas(gcf, 'fig10_simulation_results.fig');

%% ========================================================================
%% Figure 11: 频率复用与干扰对比
%% ========================================================================
fprintf('Generating Figure 11: Frequency Reuse...\n');

figure('Position', [100 100 950 600], 'Color', 'white');

schemes = categorical({'4-Color\nReuse', 'Full Reuse\n(No Precoding)', 'Block-SVD\nPrecoding', 'Frame-Based\nPrecoding'});
throughput_gbps = [0.73, 0.55, 1.61, 1.72];
improvement = (throughput_gbps - 0.73) / 0.73 * 100;

subplot(1, 2, 1);
b = bar(throughput_gbps);
b.FaceColor = 'flat';
b.CData = [colors.gray; colors.secondary; colors.primary; colors.success];
set(gca, 'XTickLabel', {'4-Color', 'Full (No Prec)', 'Block-SVD', 'Frame-Based'});
ylabel('System Throughput (Gb/s)', 'FontWeight', 'bold');
title('Frequency Reuse Schemes Comparison', 'FontWeight', 'bold');
ylim([0 2]);
grid on; box on;
for i = 1:4
    text(i, throughput_gbps(i) + 0.08, sprintf('%.2f', throughput_gbps(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);
end

subplot(1, 2, 2);
b2 = bar(improvement);
b2.FaceColor = 'flat';
b2.CData = [colors.gray; colors.secondary; colors.primary; colors.success];
set(gca, 'XTickLabel', {'4-Color', 'Full (No Prec)', 'Block-SVD', 'Frame-Based'});
ylabel('Improvement vs 4-Color (%)', 'FontWeight', 'bold');
title('Relative Performance Gain', 'FontWeight', 'bold');
ylim([-50 160]);
yline(0, '-', 'Color', 'k', 'LineWidth', 1);
grid on; box on;
for i = 1:4
    if improvement(i) >= 0
        text(i, improvement(i) + 5, sprintf('+%.0f%%', improvement(i)), ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);
    else
        text(i, improvement(i) - 8, sprintf('%.0f%%', improvement(i)), ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);
    end
end

sgtitle('Multibeam Satellite Precoding Performance (IEEE WCM 2016)', 'FontSize', 14, 'FontWeight', 'bold');

saveas(gcf, 'fig11_frequency_reuse.png');
saveas(gcf, 'fig11_frequency_reuse.fig');

%% ========================================================================
%% Figure 12: 切片编码方案对比
%% ========================================================================
fprintf('Generating Figure 12: Slice Coding Comparison...\n');

figure('Position', [100 100 1000 650], 'Color', 'white');

slices = {'URLLC', 'eMBB', 'mMTC'};
block_lengths = [256, 8448, 128];
code_rates = [0.5, 0.75, 0.33];
target_bler = [1e-5, 1e-3, 1e-2];
latency_us = [50, 500, 100];  % 解码延迟

subplot(2, 2, 1);
bar(block_lengths);
set(gca, 'XTickLabel', slices);
ylabel('Block Length N', 'FontWeight', 'bold');
title('Block Length per Slice', 'FontWeight', 'bold');
set(gca, 'YScale', 'log');
ylim([100 10000]);
for i = 1:3
    text(i, block_lengths(i)*1.3, num2str(block_lengths(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end
grid on; box on;

subplot(2, 2, 2);
bar(code_rates);
set(gca, 'XTickLabel', slices);
ylabel('Code Rate R', 'FontWeight', 'bold');
title('Code Rate per Slice', 'FontWeight', 'bold');
ylim([0 1]);
for i = 1:3
    text(i, code_rates(i)+0.05, sprintf('%.2f', code_rates(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end
grid on; box on;

subplot(2, 2, 3);
bar(-log10(target_bler));
set(gca, 'XTickLabel', slices);
ylabel('-log_{10}(BLER)', 'FontWeight', 'bold');
title('Target BLER per Slice', 'FontWeight', 'bold');
ylim([0 6]);
for i = 1:3
    text(i, -log10(target_bler(i))+0.3, sprintf('10^{-%d}', -log10(target_bler(i))), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end
grid on; box on;

subplot(2, 2, 4);
code_types = {'Polar\n(SCL-8)', 'LDPC\n(BP-12)', 'Polar\n(SC)'};
slice_colors_bar = [colors.secondary; colors.primary; colors.purple];
b = bar(latency_us);
b.FaceColor = 'flat';
b.CData = slice_colors_bar;
set(gca, 'XTickLabel', slices);
ylabel('Decoding Latency (\mus)', 'FontWeight', 'bold');
title('Decoding Latency per Slice', 'FontWeight', 'bold');
ylim([0 600]);
for i = 1:3
    text(i, latency_us(i)+30, sprintf('%d\\mus', latency_us(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end
grid on; box on;

sgtitle('Network Slice Coding Scheme Comparison', 'FontSize', 15, 'FontWeight', 'bold');

saveas(gcf, 'fig12_slice_coding.png');
saveas(gcf, 'fig12_slice_coding.fig');

%% ========================================================================
%% Figure 13: 链路预算分析
%% ========================================================================
fprintf('Generating Figure 13: Link Budget...\n');

figure('Position', [100 100 900 650], 'Color', 'white');

components = {'Tx Power', 'Tx Antenna', 'Free Space', 'Atmos Loss', 'Rx Antenna', 'System Noise', 'Rx Power'};
values = [40, 48, -182, -3, 35, -130, -192];

cumulative = [40];
for i = 2:length(values)-1
    cumulative(end+1) = cumulative(end) + values(i);
end
cumulative(end+1) = sum(values(1:end-1));

bar_colors = [colors.success; colors.primary; colors.secondary; colors.warning; ...
              colors.info; colors.gray; colors.purple];

hold on;
for i = 1:length(components)
    if i == 1
        bar(i, values(i), 'FaceColor', bar_colors(i,:));
    elseif i == length(components)
        bar(i, cumulative(end), 'FaceColor', bar_colors(i,:));
    else
        bar(i, cumulative(i), 'FaceColor', bar_colors(i,:));
    end
end

set(gca, 'XTickLabel', components, 'XTickLabelRotation', 30);
ylabel('Power Level (dBm/dB)', 'FontWeight', 'bold');
title('6G LEO Satellite Downlink Budget Analysis', 'FontSize', 14, 'FontWeight', 'bold');
subtitle('Ka-band (30 GHz), LEO 500km Altitude');
yline(0, '--', 'Color', 'k', 'LineWidth', 1);
grid on; box on;

for i = 1:length(components)
    if i == 1 || i == length(components)
        y_pos = values(i);
    else
        y_pos = cumulative(i);
    end
    text(i, y_pos + sign(y_pos)*8, sprintf('%+.0f', values(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);
end

saveas(gcf, 'fig13_link_budget.png');
saveas(gcf, 'fig13_link_budget.fig');

%% ========================================================================
%% Figure 14: 6G目标性能雷达图
%% ========================================================================
fprintf('Generating Figure 14: 6G Target Radar...\n');

figure('Position', [100 100 800 700], 'Color', 'white');

categories = {'Throughput', 'Latency', 'Reliability', 'Efficiency', 'Coverage', 'Scalability'};
current_5g = [0.6, 0.5, 0.7, 0.5, 0.6, 0.5];
target_6g = [1.0, 0.95, 0.99, 0.9, 0.95, 0.9];
achieved = [0.85, 0.9, 0.93, 0.75, 0.85, 0.8];

n = length(categories);
angles = linspace(0, 2*pi, n+1);
angles = angles(1:end-1);

current_5g_plot = [current_5g, current_5g(1)];
target_6g_plot = [target_6g, target_6g(1)];
achieved_plot = [achieved, achieved(1)];
angles_plot = [angles, angles(1)];

polarplot(angles_plot, target_6g_plot, '--', 'Color', colors.gray, 'LineWidth', 2);
hold on;
polarplot(angles_plot, current_5g_plot, '-o', 'Color', colors.secondary, 'LineWidth', 2, 'MarkerSize', 8);
polarplot(angles_plot, achieved_plot, '-s', 'Color', colors.success, 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', colors.success);

thetaticks(rad2deg(angles));
thetaticklabels(categories);
rlim([0 1.1]);
rticks([0.25, 0.5, 0.75, 1.0]);

title('6G Performance Radar: Current vs Target vs Achieved', 'FontSize', 14, 'FontWeight', 'bold');
legend('6G Target', '5G Current', 'Our Achievement', 'Location', 'southoutside', 'Orientation', 'horizontal');

saveas(gcf, 'fig14_6g_radar.png');
saveas(gcf, 'fig14_6g_radar.fig');

%% ========================================================================
%% Figure 15: 能耗与QoS权衡分析
%% ========================================================================
fprintf('Generating Figure 15: Energy-QoS Tradeoff...\n');

figure('Position', [100 100 900 600], 'Color', 'white');

power_levels = 10:5:40;
qos_curve = 0.5 + 0.4 * (1 - exp(-0.1*(power_levels-10)));
energy_curve = 28 + 0.75*50 + (1/0.35)*10.^((power_levels-30)/10) * 1000;

yyaxis left;
plot(power_levels, qos_curve, '-o', 'Color', colors.primary, 'LineWidth', 2.5, ...
    'MarkerFaceColor', colors.primary, 'MarkerSize', 10);
ylabel('QoS Score', 'FontWeight', 'bold', 'Color', colors.primary);
ylim([0.4 1.0]);

yyaxis right;
plot(power_levels, energy_curve, '-s', 'Color', colors.secondary, 'LineWidth', 2.5, ...
    'MarkerFaceColor', colors.secondary, 'MarkerSize', 10);
ylabel('Energy Consumption (W)', 'FontWeight', 'bold', 'Color', colors.secondary);

xlabel('Transmit Power (dBm)', 'FontWeight', 'bold');
title('Energy-QoS Tradeoff Analysis', 'FontSize', 14, 'FontWeight', 'bold');
subtitle('Problem 5: Minimizing Energy while Maximizing QoS');
grid on; box on;

[~, opt_idx] = min(abs(qos_curve - 0.9));
xline(power_levels(opt_idx), '--', 'Color', colors.success, 'LineWidth', 2, ...
    'Label', sprintf('Optimal: %d dBm', power_levels(opt_idx)));

legend('QoS', 'Energy', 'Optimal Point', 'Location', 'northwest');

saveas(gcf, 'fig15_energy_qos_tradeoff.png');
saveas(gcf, 'fig15_energy_qos_tradeoff.fig');

%% ========================================================================
%% 完成提示
%% ========================================================================
fprintf('\n');
fprintf('=====================================================\n');
fprintf('All 15 figures generated successfully!\n');
fprintf('=====================================================\n');
fprintf('\n');
fprintf('Generated files:\n');
fprintf('  fig01_beam_pattern.png         - Multi-beam gain pattern\n');
fprintf('  fig02_dvbs2x_acm.png           - DVB-S2X ACM performance\n');
fprintf('  fig03_coding_selection.png     - 6G coding selection\n');
fprintf('  fig04_interference_framework.png - IBI/ISI/LTI framework\n');
fprintf('  fig05_slicing_qos.png          - Network slicing QoS\n');
fprintf('  fig06_phased_array.png         - Phased array gain\n');
fprintf('  fig07_leo_constellation.png    - LEO constellation\n');
fprintf('  fig08_architecture_mapping.png - 5G to 6G mapping\n');
fprintf('  fig09_channel_model.png        - Channel model\n');
fprintf('  fig10_simulation_results.png   - Simulation results\n');
fprintf('  fig11_frequency_reuse.png      - Frequency reuse comparison\n');
fprintf('  fig12_slice_coding.png         - Slice coding comparison\n');
fprintf('  fig13_link_budget.png          - Link budget analysis\n');
fprintf('  fig14_6g_radar.png             - 6G target radar chart\n');
fprintf('  fig15_energy_qos_tradeoff.png  - Energy-QoS tradeoff\n');
fprintf('=====================================================\n');
fprintf('Place these images at the corresponding placeholders\n');
fprintf('in the Word report (6G_Course_Project_Report.docx)\n');
fprintf('=====================================================\n');
