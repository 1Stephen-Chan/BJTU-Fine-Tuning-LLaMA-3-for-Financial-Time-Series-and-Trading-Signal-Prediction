import pandas as pd
import numpy as np
import json

# --- 1. 配置 ---
WINDOW_SIZE = 5
MASK_RATIO = 0.15
VAL_START = '2024-01-02'
# 下面这个 TRADE_THRESHOLD 可作为基础参考，主要逻辑已移至 STRATEGY_PARAMS
TRADE_THRESHOLD = 0.0015 

T5_N, T10_N = 5, 10

# 核心超参数：定义不同周期的“好波段”标准
STRATEGY_PARAMS = {
    "5":  {"up_threshold": 0.015, "down_limit": -0.01, "sell_trigger": -0.015},
    "10": {"up_threshold": 0.040, "down_limit": -0.02, "sell_trigger": -0.030}
}

# --- 2. 策略逻辑函数 ---
def get_action_label(raw_closes, curr_idx, n_days, params):
    """
    基于未来窗口极值与超参数生成四分类标签
    """
    if curr_idx + n_days >= len(raw_closes) or curr_idx < 1:
        return None, None
    
    # --- A. 获取窗口数据 ---
    curr_price = raw_closes.iloc[curr_idx]
    future_window = raw_closes.iloc[curr_idx + 1 : curr_idx + n_days + 1]
    
    # --- B. 计算未来极值收益率 ---
    max_return = (future_window.max() - curr_price) / (curr_price + 1e-8)
    min_return = (future_window.min() - curr_price) / (curr_price + 1e-8)
    
    # --- C. 判定当前点是否属于 "优质波段" ---
    is_bullish_window = (max_return >= params["up_threshold"]) and \
                        (min_return >= params["down_limit"])

    # --- D. 判定 "昨天" 的表现 (用于区分 Buy 和 Stay) ---
    prev_price = raw_closes.iloc[curr_idx - 1]
    prev_window = raw_closes.iloc[curr_idx : curr_idx + n_days] 
    prev_max_r = (prev_window.max() - prev_price) / (prev_price + 1e-8)
    prev_min_r = (prev_window.min() - prev_price) / (prev_price + 1e-8)
    
    was_bullish_prev = (prev_max_r >= params["up_threshold"]) and \
                       (prev_min_r >= params["down_limit"])

    # --- E. 状态机输出 ---
    if is_bullish_window:
        if not was_bullish_prev:
            return "Buy", "空仓"    # 刚达标：进场
        else:
            return "Stay", "持仓"   # 持续达标：持有
            
    elif max_return < (params["up_threshold"] / 2) or min_return <= params["sell_trigger"]:
        return "Sell", "持仓"      # 期望太低或跌破触发线：卖出
        
    else:
        return "Wait", "空仓"      # 没肉吃或震荡：观望

# --- 3. 生成数据集 ---
def prepare_sft_with_mask():
    print(f"正在读取数据并应用 {MASK_RATIO:.0%} 随机掩码...")
    df = pd.read_csv('HistoricalData.csv', index_col='date', parse_dates=True).sort_index().ffill()
    
    # 切分训练数据
    train_df = df.loc[:VAL_START].iloc[:-1].copy()
    t_min, t_max = train_df.min(), train_df.max()
    train_norm = (train_df - t_min) / (t_max - t_min + 1e-8)
    raw_closes = train_df['close']
    cols = train_norm.columns.tolist()

    t5_list, t10_list, mixed_list = [], [], []
    limit = len(train_norm) - T10_N - WINDOW_SIZE

    for i in range(1, limit):
        curr_idx = i + WINDOW_SIZE - 1
        
        # --- 实施随机掩码生成 input_str ---
        window_data = train_norm.iloc[i : i + WINDOW_SIZE].values.copy()
        history_parts = []
        for d in range(WINDOW_SIZE):
            day_feats = []
            for j, val in enumerate(window_data[d]):
                if np.random.rand() < MASK_RATIO:
                    day_feats.append(f"{cols[j]}:[MASK]")
                else:
                    day_feats.append(f"{cols[j]}:{val:.4f}")
            history_parts.append(f"Day{d+1}:[{', '.join(day_feats)}]")
        
        input_str = " ".join(history_parts)

        # --- 遍历不同周期的任务 ---
        for n, target_list, task_name in [(T5_N, t5_list, "5"), (T10_N, t10_list, "10")]:
            # 获取对应周期的超参数
            params = STRATEGY_PARAMS[task_name]
            
            label, status = get_action_label(raw_closes, curr_idx, n, params)
            
            if label:
                # 针对“空仓”和“持仓”设计不同的专家任务
                if status == "空仓":
                    expert_task = f"评估是否存在符合(Max_Return >= {params['up_threshold']:.1%}, Max_Drawdown <= {abs(params['down_limit']):.1%})标准的建仓波段."
                else:
                    expert_task = f"评估当前持仓的利润延续性,并监控是否触及 {abs(params['sell_trigger']):.1%} 的强制风控平仓线."

                # 最终 Instruction 文本
                instruction = (
                    f"你是一位资深的量化策略专家.当前账户状态：[{status}].\n"
                    f"你的任务是:基于过去5个交易日含[MASK]噪声的数据,执行[未来{task_name}日]的波段决策预测.\n"
                    f"具体目标:{expert_task}\n"
                    f"请输出唯一的执行动作(Buy/Stay/Sell/Wait)"
                )

                entry = {
                    "instruction": instruction,
                    "input": input_str, 
                    "output": label
                }
                target_list.append(entry)
                mixed_list.append(entry)

    def save_final(data, name):
        # 统计以查看平衡情况
        actions = [d for d in data if d['output'] in ["Buy", "Stay", "Sell"]]
        waits = [d for d in data if d['output'] == "Wait"]
        
        np.random.shuffle(waits)
        balanced_waits = waits[:len(actions)]
        
        final = actions + balanced_waits
        np.random.shuffle(final)
        
        with open(f"stock_mask_{name}.jsonl", 'w', encoding='utf-8') as f:
            for d in final:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
        
        counts = {k: 0 for k in ["Buy", "Stay", "Sell", "Wait"]}
        for d in final: counts[d['output']] += 1
        print(f"{name} 数据集生成完毕！总数: {len(final)} | 详情: {counts}")

    save_final(t5_list, "T5")
    save_final(t10_list, "T10")
    save_final(mixed_list, "Mixed")

if __name__ == "__main__":
    prepare_sft_with_mask()