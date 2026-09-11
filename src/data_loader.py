"""
data_loader.py
==============
数据加载模块（参赛正式代码 v0.1 · 2026-09-08 重写）

作用：
    读取官方 sensorData zip 解压后的 collect_data*.txt，
    输出干净的传感器时间序列，供 preprocess 切窗/打标签。

真实数据格式（已分析确认）：
    txt 是 tab 分隔的 53 列宽表：
        ACC_TIME, PPG_TIME, GYRO_TIME    三传感器各自时间戳(毫秒)
        PPG1 ~ PPG44                      44 列 PPG（结构待专项破解）
        ACC_X/Y/Z, GYRO_X/Y/Z             加速度/陀螺仪
    特性：
        - ACC/GYRO：每行 = 1 个采样点(~105Hz)，时间戳每 10 行打一次(批头)
        - 行间夹大量"0 填充行"（某传感器不采样时补 0）→ 读取时筛掉
        - PPG：每 ~602ms 一个 15 行批（44 列语义待破解，v0.1 暂不处理）

模块内函数：
    load_acc_gyro(txt_path)
        读 txt → 返回 ACC/GYRO 有效采样 DataFrame

用法示例：
    from data_loader import load_acc_gyro
    acc_gyro = load_acc_gyro("collect_data28_....txt")

作者：品食Panda 队 · 假菌
"""

import pandas as pd
import zipfile


ACC_GYRO_COLS = ["ACC_TIME", "ACC_X", "ACC_Y", "ACC_Z",
                 "GYRO_X", "GYRO_Y", "GYRO_Z"]


def load_acc_gyro(txt_path):
    """
    读 sensor txt，返回 ACC/GYRO 的有效采样。

    参数
    ----
    txt_path : str
        collect_data*.txt 的路径

    返回
    ----
    pandas.DataFrame，含列: ACC_TIME, ACC_X/Y/Z, GYRO_X/Y/Z
        只保留 ACC_TIME > 0 的行（跳掉 0 填充行），索引重置
    """
    # >>> TODO(你来填) <<<
    # 提示（3 步）：
    # 1) df = pd.read_csv(txt_path, sep='\t')        # 读 53 列表
    # 2) valid = df[df['ACC_TIME'] > 0]             # 筛有效行
    # 3) return valid[ACC_GYRO_COLS].reset_index(drop=True)
    df=pd.read_csv(txt_path,sep='\t')
    valid=df[df['ACC_TIME']>0]
    return valid[ACC_GYRO_COLS].reset_index(drop=True)
def load_sensor_zip(zip_path):
    """

    :param zip_path: 直接从一个 sensorData zip 读 ACC/GYRO（不落地解压）。
    :return:返回与 load_acc_gyro 相同的 DataFrame
    """
    with zipfile.ZipFile(zip_path)as z:
        names=z.namelist()
        txt_name=[n for n in names if n.startswith("collect_data")][0]
        with z.open(txt_name)as f:
             df=pd.read_csv(f,sep='\t')
    valid=df[df['ACC_TIME']>0]
    return valid[ACC_GYRO_COLS].reset_index(drop=True)




if __name__ == "__main__":
    # ===== 自测：跑通即模块可用 =====
    demo = r"E:\workbuddy\进食检测比赛\data\_inspect\collect_data28_1785586214_1785645732.txt"
    acc_gyro = load_acc_gyro(demo)
    print("ACC/GYRO 有效采样数:", len(acc_gyro))       # 期望 ~46 万行
    print("前 3 行:")
    print(acc_gyro.head(3))
    print("列:", list(acc_gyro.columns))
