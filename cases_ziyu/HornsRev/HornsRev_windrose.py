import numpy as np
import matplotlib.pyplot as plt
import path
import matplotlib_params
import pandas as pd
# 风向频率 (0-360°，间隔30°)
directions = np.arange(0, 360, 30)
frequencies = np.array([3.8, 4.3, 5.5, 8.3, 8.7, 6.7, 8.4, 10.5, 11.5, 12.3, 13.9, 6.1]) 
frequencies=frequencies/np.sum(frequencies)


# Weibull 参数
# 62m 转换到 70m
scale = np.array([8.71, 9.36, 9.29, 10.27, 10.89, 10.49, 10.94, 11.23, 11.93, 11.94, 12.17, 10.31])*(70./62.)**0.11
shape = np.array([2.08, 2.22, 2.41, 2.37, 2.51, 2.75, 2.61, 2.51, 2.33, 2.35, 2.58, 2.01])  #shape不变

# 计算风速分布
def weibull_pdf(v, shape, scale):
    return (shape / scale) * (v / scale)**(shape - 1) * np.exp(-(v / scale)**shape)

def weibull_cdf(v, shape, scale):
    return 1. - np.exp(-(v / scale)**shape)

# 生成风速范围
wind_speeds = np.array([2,6,10,14,18,22,26])
wind_speeds_cal = 0.5*(wind_speeds[0:-1] + wind_speeds[1:])
# 计算每个方向的风速分布
wind_distribution = np.zeros((len(directions), len(wind_speeds)-1))
for i, freq in enumerate(directions):
    weibull_cdf_sum = np.sum(weibull_cdf(wind_speeds[1:], shape[i], scale[i]) - weibull_cdf(wind_speeds[0:-1], shape[i], scale[i]) )
    wind_distribution[i] = frequencies[i] * (weibull_cdf(wind_speeds[1:], shape[i], scale[i]) - weibull_cdf(wind_speeds[0:-1], shape[i], scale[i]))/weibull_cdf_sum

# 绘制风玫瑰图
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_yticks(np.arange(0,0.15,0.03))
ax.set_ylim(0,0.15)
ax.set_xticks(np.arange(0,2*np.pi,2*np.pi/12))
# 绘制每个方向的风速分布
width = np.deg2rad(30 * 1.0) 
bottom = np.zeros(len(directions))
for i in range(len(wind_speeds)-1):
    ax.bar(np.deg2rad(directions), wind_distribution[:,i],bottom=bottom,width=width, alpha=0.75)
    bottom += wind_distribution[:,i]
    # ax.fill_between(np.deg2rad(direction) * np.ones_like(wind_speeds), 0, wind_speeds, 
    #                 color='b', alpha=0.1)


# 创建输出数据
data = []
for i, wind_speed in enumerate(wind_speeds_cal):
    for j, direction in enumerate(directions):
        ws = wind_speed  
        wd = direction  # 风向
        freq_val = wind_distribution[j,i]  # 频率值
        data.append([ws, wd, freq_val])

# 转换为 DataFrame
df = pd.DataFrame(data, columns=["ws", "wd", "freq_val"])

# 保存为 CSV 文件
df.to_csv("./cases_ziyu/HornsRev/HornsRev_windrose.csv", index=False)


fmt='png'
dpi=256
load_dir = './cases_ziyu/fig/'
plt.savefig(load_dir+'HornsRev_windrose.'+fmt,dpi=dpi)