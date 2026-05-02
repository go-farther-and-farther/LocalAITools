#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 吞吐量压测 + 可视化图表生成
- 支持任意 OpenAI 兼容 API
- 自动处理非标准模型名（tiktoken fallback）
- 输出表格、百分位统计，并绘制双折线图
"""

import sys
import time
import json
import argparse
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

import requests
import numpy as np
import matplotlib.pyplot as plt

# -------------------- 配置默认值 --------------------
DEFAULT_BASE_URL = config.OPENAI_BASE_URL
DEFAULT_MODEL = config.BENCHMARK_MODEL
DEFAULT_API_KEY = config.OPENAI_API_KEY
DEFAULT_TIMEOUT = config.REQUEST_TIMEOUT_SHORT
DEFAULT_CONCURRENCY = config.DEFAULT_WORKERS
DEFAULT_OUTPUT_TOKENS = 512
FALLBACK_ENCODING = "cl100k_base"

# 默认测试的提示词长度 (tokens)
DEFAULT_TOKEN_LENGTHS = [512, 1024, 2048, 4096]

# -------------------- Token 计数与文本生成（修复版）--------------------
def generate_prompt_text(target_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    """生成接近指定 token 数量的文本（自动回退编码）"""
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding(FALLBACK_ENCODING)

        base_text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a test sentence designed to generate tokens for benchmarking purposes. "
            "It contains various common English words and punctuation. "
        )
        base_tokens = len(enc.encode(base_text))
        repeats = max(1, target_tokens // base_tokens)
        full_text = base_text * repeats
        tokens = enc.encode(full_text)[:target_tokens]
        return enc.decode(tokens)
    except ImportError:
        # 无 tiktoken 时的粗略估算
        base = "The quick brown fox jumps over the lazy dog. "
        repeats = max(1, target_tokens // 10)
        return base * repeats

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """精确计数（若可用）或估算"""
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding(FALLBACK_ENCODING)
        return len(enc.encode(text))
    except ImportError:
        return len(text) // 4

# -------------------- API 调用（流式测量）--------------------
def call_api_stream(
    prompt: str,
    base_url: str,
    model: str,
    api_key: str,
    timeout: int,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> Tuple[float, float, int, int, bool, str]:
    """
    返回：
        tftt (ms), ittl_mean (ms), prompt_tokens, completion_tokens, success, error_msg
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True
    }

    start_time = time.perf_counter()
    first_token_time = None
    token_times = []
    prompt_tokens = 0
    completion_tokens = 0
    success = False
    error_msg = ""

    try:
        with requests.post(url, headers=headers, json=payload, timeout=timeout, stream=True) as resp:
            if resp.status_code != 200:
                error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                return 0, 0, 0, 0, False, error_msg

            for line in resp.iter_lines(decode_unicode=True):
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if first_token_time is None:
                        first_token_time = time.perf_counter()

                    now = time.perf_counter()
                    token_times.append(now)

                    if "usage" in chunk:
                        usage = chunk["usage"]
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)

            end_time = time.perf_counter()

    except Exception as e:
        error_msg = str(e)
        return 0, 0, 0, 0, False, error_msg

    if first_token_time is None:
        error_msg = "No tokens received"
        return 0, 0, 0, 0, False, error_msg

    tftt = (first_token_time - start_time) * 1000

    if len(token_times) > 1:
        intervals = [(token_times[i] - token_times[i-1]) * 1000 for i in range(1, len(token_times))]
        ittl_mean = sum(intervals) / len(intervals)
    else:
        ittl_mean = 0.0

    if prompt_tokens == 0:
        prompt_tokens = count_tokens(prompt, model)
    if completion_tokens == 0:
        completion_tokens = len(token_times)

    success = True
    return tftt, ittl_mean, prompt_tokens, completion_tokens, success, error_msg

def run_single_test(
    target_tokens: int,
    base_url: str,
    model: str,
    api_key: str,
    timeout: int,
    output_tokens: int = 512,
) -> Dict:
    prompt = generate_prompt_text(target_tokens, model)
    tftt, ittl, prompt_toks, out_toks, success, err = call_api_stream(
        prompt, base_url, model, api_key, timeout, output_tokens
    )

    prefill_speed = prompt_toks / (tftt / 1000) if success and tftt > 0 else 0.0
    output_speed = 1000 / ittl if success and ittl > 0 else 0.0

    return {
        "target_tokens": target_tokens,
        "actual_prompt_tokens": prompt_toks,
        "actual_output_tokens": out_toks,
        "tftt_ms": tftt,
        "ittl_mean_ms": ittl,
        "prefill_throughput": prefill_speed,
        "output_throughput": output_speed,
        "success": success,
        "error": err if not success else ""
    }

def run_concurrent_tests(
    concurrency: int,
    target_tokens: int,
    base_url: str,
    model: str,
    api_key: str,
    timeout: int,
    output_tokens: int = 512,
) -> List[Dict]:
    results = []
    lock = threading.Lock()

    def worker():
        res = run_single_test(target_tokens, base_url, model, api_key, timeout, output_tokens)
        with lock:
            results.append(res)
        return res

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(concurrency)]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"   Worker exception: {e}")

    return results

# -------------------- 绘图函数 --------------------
def plot_results(
    token_lengths: List[int],
    prefill_vals: List[float],
    decode_vals: List[float],
    concurrency: int,
    timeout: int,
    save_path: str = "throughput_chart.png"
):
    """绘制双子图：预填充吞吐 & 输出吞吐，并标注百分位"""
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f'Throughput vs Prompt Length (Concurrency={concurrency}, Timeout={timeout}s)',
                 fontsize=14, fontweight='bold')

    # 百分位数
    p_pre = np.percentile(prefill_vals, [50, 90, 95])
    p_out = np.percentile(decode_vals, [50, 90, 95])

    # 上图：预填充
    ax1.plot(token_lengths, prefill_vals, marker='o', linestyle='-', linewidth=2,
             color='steelblue', label='Prefill Throughput')
    ax1.axhline(p_pre[0], color='gray', linestyle='--', alpha=0.7, label=f"P50: {p_pre[0]:.1f}")
    ax1.axhline(p_pre[1], color='orange', linestyle='--', alpha=0.7, label=f"P90: {p_pre[1]:.1f}")
    ax1.axhline(p_pre[2], color='red', linestyle='--', alpha=0.7, label=f"P95: {p_pre[2]:.1f}")
    ax1.set_ylabel('Prefill Throughput (tokens/s)')
    ax1.set_yscale('log')
    ax1.grid(True, which='both', linestyle=':', alpha=0.6)
    ax1.legend(loc='best')
    textstr = f'Range: {min(prefill_vals):.0f} - {max(prefill_vals):.0f}\nMean: {np.mean(prefill_vals):.0f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.98, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    # 下图：输出
    ax2.plot(token_lengths, decode_vals, marker='s', linestyle='-', linewidth=2,
             color='darkgreen', label='Decode Throughput')
    ax2.axhline(p_out[0], color='gray', linestyle='--', alpha=0.7, label=f"P50: {p_out[0]:.1f}")
    ax2.axhline(p_out[1], color='orange', linestyle='--', alpha=0.7, label=f"P90: {p_out[1]:.1f}")
    ax2.axhline(p_out[2], color='red', linestyle='--', alpha=0.7, label=f"P95: {p_out[2]:.1f}")
    ax2.set_xlabel('Prompt Length (tokens)')
    ax2.set_ylabel('Decode Throughput (tokens/s)')
    ax2.set_xscale('log', base=2)
    ax2.grid(True, which='both', linestyle=':', alpha=0.6)
    ax2.legend(loc='best')
    textstr2 = f'Range: {min(decode_vals):.0f} - {max(decode_vals):.0f}\nMean: {np.mean(decode_vals):.0f}'
    ax2.text(0.98, 0.95, textstr2, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    ax2.set_xticks(token_lengths)
    ax2.set_xticklabels([str(p) for p in token_lengths], rotation=45, ha='right')
    ax2.set_xlim(token_lengths[0], token_lengths[-1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"📈 图表已保存: {save_path}")

# -------------------- 主流程 --------------------
def run_benchmark(
    token_lengths: List[int],
    concurrency: int,
    base_url: str,
    model: str,
    api_key: str,
    timeout: int,
    output_tokens: int = 512,
    save_json: str = "benchmark_results.json",
    save_plot: str = "throughput_chart.png"
):
    print(f"\n{'='*80}")
    print(f"🚀 LLM 性能压测")
    print(f"📡 API: {base_url} | 模型: {model} | 并发: {concurrency} | 超时: {timeout}s")
    print(f"📏 测试长度: {token_lengths}")
    print(f"{'='*80}\n")

    all_stats = []   # 汇总每个长度平均值
    detailed_results = []  # 保存所有原始结果

    for length in token_lengths:
        print(f"🔍 测试提示词长度: {length} tokens ...")
        results = run_concurrent_tests(
            concurrency, length, base_url, model, api_key, timeout, output_tokens
        )
        detailed_results.append({"target_tokens": length, "results": results})

        successes = [r for r in results if r["success"]]
        if not successes:
            print(f"   ❌ 全部失败，跳过统计")
            all_stats.append({
                "length": length,
                "tftt": None, "ittl": None, "prefill": None, "output": None,
                "status": f"0/{concurrency} failed"
            })
            continue

        avg_tftt = np.mean([r["tftt_ms"] for r in successes])
        avg_ittl = np.mean([r["ittl_mean_ms"] for r in successes])
        avg_prefill = np.mean([r["prefill_throughput"] for r in successes])
        avg_output = np.mean([r["output_throughput"] for r in successes])

        print(f"   ✅ 成功 {len(successes)}/{concurrency}, TFTT={avg_tftt:.2f}ms, ITTL={avg_ittl:.2f}ms, "
              f"预填充={avg_prefill:.2f} tok/s, 输出={avg_output:.2f} tok/s")

        all_stats.append({
            "length": length,
            "tftt": avg_tftt,
            "ittl": avg_ittl,
            "prefill": avg_prefill,
            "output": avg_output,
            "status": f"{len(successes)}/{concurrency} success"
        })

        time.sleep(0.5)

    # ---------- 保存 JSON ----------
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump({"summary": all_stats, "details": detailed_results}, f, indent=2)
    print(f"\n💾 详细结果已保存至: {save_json}")

    # ---------- 输出表格 ----------
    print("\n" + "="*80)
    print("📊 测试结果汇总")
    print("="*80)
    print(f"{'提示词长度':<12} {'TFTT(ms)':<12} {'ITTL平均(ms)':<12} {'预填充速度(tok/s)':<18} {'输出速度(tok/s)':<16} {'状态':<15}")
    print("-"*80)

    for stat in all_stats:
        if stat["tftt"] is None:
            print(f"{stat['length']:<12} {'-':<12} {'-':<12} {'-':<18} {'-':<16} {stat['status']:<15}")
        else:
            print(f"{stat['length']:<12} {stat['tftt']:<12.2f} {stat['ittl']:<12.2f} "
                  f"{stat['prefill']:<18.2f} {stat['output']:<16.2f} {stat['status']:<15}")

    # ---------- 统计摘要 ----------
    prefill_vals = [s["prefill"] for s in all_stats if s["prefill"] is not None]
    output_vals = [s["output"] for s in all_stats if s["output"] is not None]

    if prefill_vals:
        print("\n--- 总吞吐量性能统计 ---")
        print(f"总预填充吞吐范围: {min(prefill_vals):.2f} - {max(prefill_vals):.2f} tokens/s")
        print(f"总输出吞吐范围:    {min(output_vals):.2f} - {max(output_vals):.2f} tokens/s")
        print(f"平均总预填充吞吐:  {np.mean(prefill_vals):.2f} tokens/s")
        print(f"平均总输出吞吐:    {np.mean(output_vals):.2f} tokens/s")

        print("\n--- 百分比统计 (P50/P90/P95) ---")
        p_pre = np.percentile(prefill_vals, [50, 90, 95])
        p_out = np.percentile(output_vals, [50, 90, 95])
        print(f"Prefill吞吐: P50={p_pre[0]:.2f}  P90={p_pre[1]:.2f}  P95={p_pre[2]:.2f} tokens/s")
        print(f"Decode吞吐:  P50={p_out[0]:.2f}  P90={p_out[1]:.2f}  P95={p_out[2]:.2f} tokens/s")

        # ---------- 绘制图表 ----------
        valid_lengths = [s["length"] for s in all_stats if s["prefill"] is not None]
        plot_results(
            token_lengths=valid_lengths,
            prefill_vals=prefill_vals,
            decode_vals=output_vals,
            concurrency=concurrency,
            timeout=timeout,
            save_path=save_plot
        )
    else:
        print("\n⚠️ 没有成功的测试数据，无法绘图。")

    print("\n✅ 测试完成。")

# -------------------- 命令行入口 --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 吞吐量压测 + 图表生成")
    parser.add_argument("--url", type=str, default=DEFAULT_BASE_URL, help="API Base URL")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL, help="模型名称")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="API Key")
    parser.add_argument("--concurrency", "-c", type=int, default=DEFAULT_CONCURRENCY, help="并发数")
    parser.add_argument("--timeout", "-t", type=int, default=DEFAULT_TIMEOUT, help="请求超时(秒)")
    parser.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS, help="每次请求生成的 token 数")
    parser.add_argument("--lengths", type=str, help="逗号分隔的测试长度列表，例如 512,1024,2048")
    parser.add_argument("--max-length", type=int, help="只测试到指定最大长度（使用预设列表）")
    parser.add_argument("--save-json", type=str, default=str(config.OUTPUT_DIR / "benchmarks" / "benchmark_results.json"), help="保存原始数据的 JSON 路径")
    parser.add_argument("--save-plot", type=str, default=str(config.OUTPUT_DIR / "benchmarks" / "throughput_chart.png"), help="保存图表的 PNG 路径")
    args = parser.parse_args()

    # 确定测试长度列表
    if args.lengths:
        lengths = [int(x.strip()) for x in args.lengths.split(",")]
    else:
        lengths = DEFAULT_TOKEN_LENGTHS
        if args.max_length:
            lengths = [l for l in lengths if l <= args.max_length]

    run_benchmark(
        token_lengths=lengths,
        concurrency=args.concurrency,
        base_url=args.url,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        output_tokens=args.output_tokens,
        save_json=args.save_json,
        save_plot=args.save_plot,
    )