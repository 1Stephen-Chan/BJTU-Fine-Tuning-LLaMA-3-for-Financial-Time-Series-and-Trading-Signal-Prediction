import pandas as pd
import numpy as np
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from modelscope import snapshot_download
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 配置参数 ---
WINDOW_SIZE = 5
MODEL_ID = 'LLM-Research/Llama-3.2-1B-Instruct'
LORA_PATH = "./llama_3_2_stock_ft_mixed"
VAL_START, VAL_END = '2024-01-02', '2025-12-15'

# 买卖频率控制参数
TRADE_THRESHOLD = 0.009

# --- 数据全流程预处理 ---
def prepare_data():
    print("Step 1: 正在读入数据")
    df = pd.read_csv('HistoricalData.csv', index_col='date', parse_dates=True).sort_index().ffill()
    val_df = df.loc[VAL_START : VAL_END].copy()
    train_df = df.drop(val_df.index).copy()
    
    t_min, t_max = train_df.min(), train_df.max()
    val_norm = (val_df - t_min) / (t_max - t_min + 1e-8)

    def create_windows(data_norm):
        x = data_norm.values
        return np.array([x[i : i + WINDOW_SIZE] for i in range(len(x) - WINDOW_SIZE + 1)])

    X_val = create_windows(val_norm)
    return X_val, val_df, train_df.columns.tolist()

# --- 交易回测 ---
def run_real_world_backtest(X_val, val_df, cols):
    raw_dir = snapshot_download(MODEL_ID, cache_dir='./')
    base_path = os.path.abspath(raw_dir)
    
    print("Step 2: 正在加载模型")
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path, dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH).eval()

    balance = 1.0
    position = 0
    history_balance = [1.0]
    actions_history = []

    print("\n" + "-" * 20 + f" 交易流水账 (阈值: {TRADE_THRESHOLD:.2%}) " + "-" * 20)
    print(f"{'交易日期':<12} | {'模型预测':<10} | {'动作':<12} | {'当日盈亏':<10} | {'账户净值'}")
    print("-" * 85)

    for i in tqdm(range(len(X_val) - 1)):
        # 构建历史窗口字符串
        history_str = " ".join([
            f"Day{d+1}: [{', '.join([f'{cols[j]}:{X_val[i][d][j]:.4f}' for j in range(len(cols))])}]"
            for d in range(WINDOW_SIZE)
        ])
        
        prompt = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                  f"你是一个专业的股票分析助手。<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
                  f"数据如下：{history_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                  f"Next_Close:")

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10, temperature=0.1, pad_token_id=tokenizer.eos_token_id)
        
        resp = tokenizer.decode(out[0], skip_special_tokens=True).split("assistant")[-1].replace("Next_Close:", "").strip().lower()
        
        # 默认持有
        signal = "hold"
        try:
            pred_num = float(resp.split()[0])
            curr_price = X_val[i][-1][0]
            
            # 使用 TRADE_THRESHOLD 控制交易频率
            if pred_num > curr_price * (1 + TRADE_THRESHOLD):
                signal = "buy"
            elif pred_num < curr_price * (1 - TRADE_THRESHOLD):
                signal = "sell"
            else:
                signal = "hold"
        except:
            if "buy" in resp: signal = "buy"
            elif "sell" in resp: signal = "sell"
            else: signal = "hold"

        curr_stock_price_val = val_df['close'].iloc[i + WINDOW_SIZE - 1]
        next_stock_price_val = val_df['close'].iloc[i + WINDOW_SIZE]
        market_return = (next_stock_price_val - curr_stock_price_val) / (curr_stock_price_val + 1e-8)
        
       # 动作
        action_text = "观望"
        current_action_code = 0

        if signal == "buy":
            if position == 0:
                # 开仓买入
                action_text = "🔴 买入开仓"
                position = 1
                current_action_code = 1
            else:
                # 已经持仓，继续持有
                action_text = "持有"
                current_action_code = 0
        elif signal == "sell":
            if position == 1:
                # 平仓卖出
                action_text = "🟢 卖出平仓"
                position = 0
                current_action_code = -1
            else:
                # 已经空仓，继续观望
                action_text = "观望"
                current_action_code = 0
        else:
            # signal == "hold"
            if position == 1:
                action_text = "持有"
            else:
                action_text = "观望"
            current_action_code = 0


        daily_profit = position * market_return
        balance *= (1 + daily_profit)
        history_balance.append(balance)
        actions_history.append(current_action_code)

        # 每日打印交易流水
        date_label = str(val_df.index[i + WINDOW_SIZE].date())
        print(f"{date_label:<12} | {resp[:10]:<10} | {action_text:<12} | {daily_profit:>10.2%} | {balance:>10.4f}")

    print("-" * 85)
    final_return = (balance - 1)
    print(f"回测结束:最终资产: {balance:.4f} (总收益: {final_return:.2%})")

    # --- 可视化 ---
    display_len = len(actions_history)
    plot_dates = val_df.index[WINDOW_SIZE : WINDOW_SIZE + display_len]
    plot_price = val_df['close'].iloc[WINDOW_SIZE : WINDOW_SIZE + display_len].values
    plot_actions = actions_history
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})

    # 股价与买卖标记
    ax1.plot(plot_dates, plot_price, color='black', linewidth=1, alpha=0.6, label='Close Price')
    for i in range(len(plot_actions)):
        if plot_actions[i] == 1:
            ax1.scatter(plot_dates[i], plot_price[i], color='red', marker='^', s=80)
            ax1.text(plot_dates[i], plot_price[i]*1.002, 'B', color='red', fontweight='bold')
        elif plot_actions[i] == -1:
            ax1.scatter(plot_dates[i], plot_price[i], color='green', marker='v', s=80)
            ax1.text(plot_dates[i], plot_price[i]*0.998, 'S', color='green', fontweight='bold')

    ax1.set_title(f"AI Strategy Execution (Threshold: {TRADE_THRESHOLD:.2%}, Return: {final_return:.2%})")
    ax1.set_ylabel('Stock Price')
    ax1.grid(True, alpha=0.3)

    # 累计净值
    plot_equity = history_balance[1:]
    ax2.fill_between(plot_dates, 1.0, plot_equity, color='blue', alpha=0.1)
    ax2.plot(plot_dates, plot_equity, color='blue', label='Strategy Net Worth')
    ax2.axhline(y=1.0, color='gray', linestyle='--')
    ax2.set_ylabel('Equity')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"trading_analysis_controlled_{TRADE_THRESHOLD:.4f}.png")
    print(f"图表已保存,当前交易频率阈值: {TRADE_THRESHOLD}")
    plt.show()

if __name__ == "__main__":
    X_val, val_df, columns = prepare_data()
    run_real_world_backtest(X_val, val_df, columns)
