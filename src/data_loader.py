"""
data_loader.py
==============
数据加载模块。

作用：
    把原始传感器数据（CSV / 官方数据集）读进来，变成程序能处理的格式。
    官方数据 8 月中旬在 HUAWEI Research 平台公布，现在先用公开数据集（UCI HAR / WESAD）练手。

将来要写的函数（现在先留空，只写注释）：
    - load_csv(path): 读一个 CSV 文件，返回 pandas.DataFrame
    - get_sensor_columns(df): 从 DataFrame 里挑出 IMU / PPG 相关列
    - split_train_test(df, ...): 划分训练集 / 测试集

设计原则：
    这个函数要和具体数据格式解耦 —— 等官方数据一来，只改这里，
    后面的 preprocess / detect / evaluate 都不用动。
"""

# TODO: 实现 load_csv
# TODO: 实现 get_sensor_columns
# TODO: 实现 split_train_test
