#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 吞吐量压测 + 专业报告图表生成
- 支持任意 OpenAI 兼容 API
- 自动处理非标准模型名（tiktoken fallback）
- 输出表格、百分位统计，生成 2x2 专业报告图
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# -------------------- 配置默认值 --------------------
DEFAULT_BASE_URL = config.OPENAI_BASE_URL
DEFAULT_MODEL = config.BENCHMARK_MODEL
DEFAULT_API_KEY = config.OPENAI_API_KEY
DEFAULT_TIMEOUT = config.REQUEST_TIMEOUT_SHORT
DEFAULT_CONCURRENCY = config.DEFAULT_WORKERS
DEFAULT_OUTPUT_TOKENS = 512
FALLBACK_ENCODING = "cl100k_base"

_stop_flag = threading.Event()

def request_stop():
    """请求停止当前正在执行的压测任务"""
    _stop_flag.set()

# 默认测试的提示词长度 (tokens)
DEFAULT_TOKEN_LENGTHS = [512, 1024, 2048, 4096]

# -------------------- 中文字体配置 --------------------
def _setup_chinese_font():
    """自动检测并配置中文字体"""
    _available = {f.name for f in fm.fontManager.ttflist}
    for _fc in ['Microsoft YaHei', 'SimHei', 'STSong', 'FangSong', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']:
        if _fc in _available:
            plt.rcParams['font.sans-serif'] = [_fc, 'DejaVu Sans']
            return _fc
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    return 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False

# -------------------- Token 计数与文本生成 --------------------
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
) -> Tuple[float, float, int, int, bool, str, List[float]]:
    """
    返回：
        tftt (ms), ittl_mean (ms), prompt_tokens, completion_tokens, success, error_msg, token_intervals
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
                return 0, 0, 0, 0, False, error_msg, []

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
        return 0, 0, 0, 0, False, error_msg, []

    if first_token_time is None:
        error_msg = "No tokens received"
        return 0, 0, 0, 0, False, error_msg, []

    tftt = (first_token_time - start_time) * 1000

    intervals = []
    if len(token_times) > 1:
        intervals = [(token_times[i] - token_times[i-1]) * 1000 for i in range(1, len(token_times))]
    ittl_mean = sum(intervals) / len(intervals) if intervals else 0.0

    if prompt_tokens == 0:
        prompt_tokens = count_tokens(prompt, model)
    if completion_tokens == 0:
        completion_tokens = len(token_times)

    success = True
    return tftt, ittl_mean, prompt_tokens, completion_tokens, success, error_msg, intervals

def run_single_test(
    target_tokens: int,
    base_url: str,
    model: str,
    api_key: str,
    timeout: int,
    output_tokens: int = 512,
) -> Dict:
    prompt = generate_prompt_text(target_tokens, model)
    tftt, ittl, prompt_toks, out_toks, success, err, intervals = call_api_stream(
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
        "error": err if not success else "",
        "intervals": intervals
    }

def run_concurrent_tests(
    concurrency: int,
    target_tokens: int,
    base_url: str,
    model: str,
    api_key: str,
    timeout: int,
    output_tokens: int = 512,
    progress_callback=None,
) -> List[Dict]:
    results = []
    lock = threading.Lock()
    done_count = [0]

    def worker():
        res = run_single_test(target_tokens, base_url, model, api_key, timeout, output_tokens)
        with lock:
            results.append(res)
            done_count[0] += 1
            if progress_callback:
                progress_callback(done_count[0], concurrency)
        return res

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(concurrency)]
        for future in as_completed(futures):
            if _stop_flag.is_set():
                executor.shutdown(wait=False, cancel_futures=True)
                print("   ⏹️ 已请求停止")
                break
            try:
                future.result()
            except Exception as e:
                print(f"   Worker exception: {e}")

    return results

# ==================== 专业报告图表 ====================
def plot_results(
    detailed_results: List[Dict],
    concurrency: int,
    timeout: int,
    model_name: str = "",
    total_duration: float = 0.0,
    save_path: str = "throughput_chart.png"
):
    """生成 2x2 专业性能报告图：TTFT百分位、输出吞吐量、ITTL、输出稳定性"""
    _setup_chinese_font()

    # ---- 收集数据 ----
    token_lengths = []
    ttft_data = []      # 每个长度的所有成功请求 TTFT
    ittl_data = []      # 每个长度的所有成功请求 ITTL
    prefill_data = []   # 每个长度的平均预填充吞吐
    decode_data = []    # 每个长度的所有成功请求 输出吞吐
    all_intervals = []  # 每个长度的所有 ITTL 间隔

    for entry in detailed_results:
        successes = [r for r in entry["results"] if r["success"]]
        if not successes:
            continue
        token_lengths.append(entry["target_tokens"])
        ttft_data.append([r["tftt_ms"] for r in successes])
        ittl_data.append([r["ittl_mean_ms"] for r in successes])
        prefill_data.append([r["prefill_throughput"] for r in successes])
        decode_data.append([r["output_throughput"] for r in successes])
        entry_intervals = []
        for r in successes:
            entry_intervals.extend(r.get("intervals", []))
        all_intervals.append(entry_intervals if entry_intervals else [0])

    if not token_lengths:
        print("⚠️ 没有成功的测试数据，无法绘图")
        return

    # ---- 样式配置 ----
    try:
        plt.style.use('ggplot')
    except Exception:
        pass

    # 配色方案
    C_RED = '#E74C3C'
    C_BLUE = '#3498DB'
    C_GREEN = '#27AE60'
    C_PURPLE = '#8E44AD'
    C_ORANGE = '#F39C12'
    C_DARK = '#2C3E50'
    C_GRAY = '#95A5A6'
    C_BG = '#FAFAFA'

    fig = plt.figure(figsize=(16, 11), facecolor='white')

    # ---- 顶部标题区域 ----
    ax_title = fig.add_axes([0, 0.92, 1, 0.08], facecolor='white')
    ax_title.axis('off')

    # 获取实际输出 token 数
    out_tok = "?"
    for entry in detailed_results:
        for r in entry["results"]:
            if r["success"]:
                out_tok = str(r.get("actual_output_tokens", "?"))
                break
        if out_tok != "?":
            break

    # 获取实际并发成功数
    total_success = sum(
        sum(1 for r in entry["results"] if r["success"])
        for entry in detailed_results
    )
    total_tests = sum(len(entry["results"]) for entry in detailed_results)

    from datetime import datetime
    test_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration_str = f"{total_duration:.0f}秒" if total_duration > 0 else "N/A"

    # 标题
    ax_title.text(0.5, 0.75, '大模型推理性能测试报告', fontsize=22, fontweight='bold',
                  color=C_DARK, ha='center', va='center', transform=ax_title.transAxes)

    # 副标题信息
    subtitle_parts = []
    if model_name:
        subtitle_parts.append(f'模型: {model_name}')
    subtitle_parts.append(f'测试时间: {test_time}')
    subtitle_parts.append(f'测试耗时: {duration_str}')
    subtitle = '    |    '.join(subtitle_parts)
    ax_title.text(0.5, 0.35, subtitle, fontsize=10, color=C_GRAY,
                  ha='center', va='center', transform=ax_title.transAxes)

    # 分隔线
    ax_title.axhline(y=0.05, xmin=0.05, xmax=0.95, color=C_BLUE, linewidth=2, alpha=0.6)

    # ---- 2x2 图表区域 ----
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.95, top=0.88, bottom=0.08,
                          hspace=0.32, wspace=0.28)

    # ========== 左上: TTFT 首Token延迟（百分位柱状图） ==========
    ax1 = fig.add_subplot(gs[0, 0])
    p50 = [np.percentile(d, 50) for d in ttft_data]
    p90 = [np.percentile(d, 90) for d in ttft_data]
    p95 = [np.percentile(d, 95) for d in ttft_data]

    x = np.arange(len(token_lengths))
    bar_w = 0.25
    bars1 = ax1.bar(x - bar_w, p50, bar_w, label='P50', color=C_GREEN, alpha=0.85, edgecolor='white', linewidth=0.5)
    bars2 = ax1.bar(x, p90, bar_w, label='P90', color=C_ORANGE, alpha=0.85, edgecolor='white', linewidth=0.5)
    bars3 = ax1.bar(x + bar_w, p95, bar_w, label='P95', color=C_RED, alpha=0.85, edgecolor='white', linewidth=0.5)

    # 柱子上方标注数值
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, h + max(p95)*0.01,
                        f'{h:.0f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax1.set_title('① 首 Token 延迟 (TTFT)', fontsize=12, fontweight='bold', color=C_DARK, pad=10)
    ax1.set_ylabel('TTFT (ms)', fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{t}' for t in token_lengths], fontsize=8)
    ax1.set_xlabel('输入长度 (tokens)', fontsize=9)
    ax1.legend(fontsize=8, loc='upper left', framealpha=0.8)
    ax1.grid(True, axis='y', linestyle=':', alpha=0.3)
    ax1.set_facecolor(C_BG)

    # ========== 右上: 输出吞吐量（柱状图 + 均值线） ==========
    ax2 = fig.add_subplot(gs[0, 1])
    means_decode = [np.mean(d) for d in decode_data]
    stds_decode = [np.std(d) for d in decode_data]

    bars = ax2.bar(x, means_decode, 0.5, color=C_BLUE, alpha=0.85, edgecolor='white', linewidth=0.5)

    # 误差线
    ax2.errorbar(x, means_decode, yerr=stds_decode, fmt='none', ecolor=C_DARK,
                elinewidth=1.5, capsize=4, capthick=1.2, alpha=0.6)

    # 柱子上方标注
    for i, (bar, m, s) in enumerate(zip(bars, means_decode, stds_decode)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(means_decode)*0.02,
                f'{m:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color=C_BLUE)

    # 均值参考线
    avg_decode = np.mean(means_decode)
    ax2.axhline(avg_decode, color=C_RED, linestyle='--', alpha=0.6, linewidth=1.2)
    ax2.text(len(token_lengths)-0.5, avg_decode + max(means_decode)*0.02,
            f'平均: {avg_decode:.1f} tok/s', fontsize=8, color=C_RED, ha='right',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=C_RED))

    ax2.set_title('② 输出吞吐量', fontsize=12, fontweight='bold', color=C_DARK, pad=10)
    ax2.set_ylabel('吞吐量 (tok/s)', fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{t}' for t in token_lengths], fontsize=8)
    ax2.set_xlabel('输入长度 (tokens)', fontsize=9)
    ax2.grid(True, axis='y', linestyle=':', alpha=0.3)
    ax2.set_facecolor(C_BG)

    # ========== 左下: 预填充吞吐量（散点 + 连线） ==========
    ax3 = fig.add_subplot(gs[1, 0])
    means_prefill = [np.mean(d) for d in prefill_data]
    stds_prefill = [np.std(d) for d in prefill_data]
    mins_prefill = [np.min(d) for d in prefill_data]
    maxs_prefill = [np.max(d) for d in prefill_data]

    # 范围带
    ax3.fill_between(x, mins_prefill, maxs_prefill, alpha=0.12, color=C_GREEN)
    # 散点
    for i, (xi, ys) in enumerate(zip(x, prefill_data)):
        if len(ys) > 1:
            jitter = np.random.default_rng(42).normal(0, 0.06, len(ys))
            ax3.scatter([xi]*len(ys) + jitter, ys, alpha=0.45, s=20, color=C_GREEN, edgecolors='none', zorder=3)
        else:
            ax3.scatter(xi, ys[0], alpha=0.7, s=30, color=C_GREEN, edgecolors='white', linewidth=0.5, zorder=3)
    # 均值连线
    ax3.plot(x, means_prefill, marker='D', linewidth=2.2, color=C_GREEN,
            markersize=6, markeredgecolor='white', markeredgewidth=0.8, zorder=4)
    # 标注
    for xi, m, s in zip(x, means_prefill, stds_prefill):
        ax3.annotate(f'{m:.0f}±{s:.0f}', (xi, m), textcoords="offset points",
                    xytext=(0, -14), ha='center', fontsize=7, color=C_GREEN, fontweight='bold')

    ax3.set_title('③ 预填充吞吐量 (Prefill)', fontsize=12, fontweight='bold', color=C_DARK, pad=10)
    ax3.set_ylabel('Prefill (tok/s)', fontsize=9)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{t}' for t in token_lengths], fontsize=8)
    ax3.set_xlabel('输入长度 (tokens)', fontsize=9)
    ax3.grid(True, linestyle=':', alpha=0.3)
    ax3.set_facecolor(C_BG)

    # ========== 右下: 输出稳定性（箱线图） ==========
    ax4 = fig.add_subplot(gs[1, 1])

    # 使用箱线图展示每个长度的输出吞吐分布
    bp = ax4.boxplot(decode_data, positions=x, widths=0.4, patch_artist=True,
                     boxprops=dict(facecolor=C_PURPLE, alpha=0.4, linewidth=1.2),
                     whiskerprops=dict(color=C_PURPLE, linewidth=1.2),
                     capprops=dict(color=C_PURPLE, linewidth=1.2),
                     medianprops=dict(color=C_RED, linewidth=2),
                     flierprops=dict(marker='o', markerfacecolor=C_PURPLE, markersize=4, alpha=0.5))

    # 叠加散点
    for i, (xi, ys) in enumerate(zip(x, decode_data)):
        if len(ys) > 1:
            jitter = np.random.default_rng(42).normal(0, 0.06, len(ys))
            ax4.scatter([xi]*len(ys) + jitter, ys, alpha=0.4, s=18, color=C_PURPLE, edgecolors='none', zorder=3)
        else:
            ax4.scatter(xi, ys[0], alpha=0.6, s=25, color=C_PURPLE, edgecolors='white', linewidth=0.5, zorder=3)

    # CV% 标注
    for i, (xi, ys) in enumerate(zip(x, decode_data)):
        if len(ys) > 1 and np.mean(ys) > 0:
            cv = np.std(ys) / np.mean(ys) * 100
            ax4.text(xi, max(ys) + max(max(d) for d in decode_data)*0.03,
                    f'CV:{cv:.1f}%', ha='center', fontsize=7, color=C_PURPLE, fontweight='bold')

    ax4.set_title('④ 输出稳定性', fontsize=12, fontweight='bold', color=C_DARK, pad=10)
    ax4.set_ylabel('输出吞吐量 (tok/s)', fontsize=9)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{t}' for t in token_lengths], fontsize=8)
    ax4.set_xlabel('输入长度 (tokens)', fontsize=9)
    ax4.grid(True, axis='y', linestyle=':', alpha=0.3)
    ax4.set_facecolor(C_BG)

    # ---- 底部汇总统计栏 ----
    ax_footer = fig.add_axes([0, 0.0, 1, 0.06], facecolor='white')
    ax_footer.axis('off')

    # 计算汇总
    all_ttft = [t for d in ttft_data for t in d]
    all_ittl = [t for d in ittl_data for t in d]
    all_decode = [t for d in decode_data for t in d]
    all_prefill = [t for d in prefill_data for t in d]

    if all_ttft:
        summary = (
            f'📊 汇总:  并发={concurrency}  |  '
            f'TTFT P50={np.percentile(all_ttft,50):.0f}ms P90={np.percentile(all_ttft,90):.0f}ms P95={np.percentile(all_ttft,95):.0f}ms  |  '
            f'输出吞吐 avg={np.mean(all_decode):.1f} tok/s  |  '
            f'预填充 avg={np.mean(all_prefill):.0f} tok/s  |  '
            f'成功率 {total_success}/{total_tests}'
        )
        ax_footer.text(0.5, 0.6, summary, fontsize=9, color=C_DARK,
                      ha='center', va='center', transform=ax_footer.transAxes,
                      bbox=dict(boxstyle='round,pad=0.4', facecolor='#ECF0F1', alpha=0.8, edgecolor=C_BLUE))

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"📈 报告图表已保存: {save_path}")

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
    save_plot: str = "throughput_chart.png",
    progress_callback=None,
):
    print(f"\n{'='*80}")
    print(f"🚀 LLM 性能压测")
    print(f"📡 API: {base_url} | 模型: {model} | 并发: {concurrency} | 超时: {timeout}s")
    print(f"📏 测试长度: {token_lengths}")
    print(f"{'='*80}\n")

    _stop_flag.clear()
    Path(save_json).parent.mkdir(parents=True, exist_ok=True)
    bench_start = time.time()
    all_stats = []
    detailed_results = []

    for length in token_lengths:
        if _stop_flag.is_set():
            print("\n⏹️ 已请求停止压测")
            break
        print(f"🔍 测试提示词长度: {length} tokens ...")
        results = run_concurrent_tests(
            concurrency, length, base_url, model, api_key, timeout, output_tokens,
            progress_callback=progress_callback,
        )
        detailed_results.append({"target_tokens": length, "results": results})

        successes = [r for r in results if r["success"]]
        if not successes:
            print(f"   ❌ 全部失败，跳过统计")
            all_stats.append({
                "length": length,
                "tftt": None, "ittl": None, "prefill": None, "output": None,
                "tftt_p50": None, "tftt_p90": None, "tftt_p95": None,
                "status": f"0/{concurrency} failed"
            })
            continue

        ttfts = [r["tftt_ms"] for r in successes]
        ittls = [r["ittl_mean_ms"] for r in successes]
        prefills = [r["prefill_throughput"] for r in successes]
        outputs = [r["output_throughput"] for r in successes]

        avg_tftt = np.mean(ttfts)
        avg_ittl = np.mean(ittls)
        avg_prefill = np.mean(prefills)
        avg_output = np.mean(outputs)

        print(f"   ✅ 成功 {len(successes)}/{concurrency}, TTFT={avg_tftt:.2f}ms, ITTL={avg_ittl:.2f}ms, "
              f"预填充={avg_prefill:.2f} tok/s, 输出={avg_output:.2f} tok/s")

        all_stats.append({
            "length": length,
            "tftt": avg_tftt,
            "ittl": avg_ittl,
            "prefill": avg_prefill,
            "output": avg_output,
            "tftt_p50": np.percentile(ttfts, 50),
            "tftt_p90": np.percentile(ttfts, 90),
            "tftt_p95": np.percentile(ttfts, 95),
            "output_std": np.std(outputs),
            "status": f"{len(successes)}/{concurrency} success"
        })

        time.sleep(0.5)

    total_duration = time.time() - bench_start

    # ---------- 保存 JSON ----------
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump({
            "summary": all_stats,
            "details": detailed_results,
            "config": {
                "model": model, "base_url": base_url, "concurrency": concurrency,
                "timeout": timeout, "output_tokens": output_tokens,
                "token_lengths": token_lengths, "total_duration": total_duration
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"\n💾 详细结果已保存至: {save_json}")

    # ---------- 输出表格 ----------
    print("\n" + "="*100)
    print("📊 测试结果汇总")
    print("="*100)
    print(f"{'输入长度':<10} {'输出长度':<10} {'并发':<6} {'Prefill(tok/s)':<16} {'吞吐量(tok/s)':<16} "
          f"{'TTFT(ms)':<12} {'ITL(ms)':<12} {'状态':<15}")
    print("-"*100)

    for stat in all_stats:
        if stat["tftt"] is None:
            print(f"{stat['length']:<10} {'-':<10} {concurrency:<6} {'-':<16} {'-':<16} "
                  f"{'-':<12} {'-':<12} {stat['status']:<15}")
        else:
            print(f"{stat['length']:<10} {output_tokens:<10} {concurrency:<6} "
                  f"{stat['prefill']:<16.2f} {stat['output']:<16.2f} "
                  f"{stat['tftt']:<12.2f} {stat['ittl']:<12.2f} {stat['status']:<15}")

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
        plot_results(
            detailed_results=detailed_results,
            concurrency=concurrency,
            timeout=timeout,
            model_name=model,
            total_duration=total_duration,
            save_path=save_plot
        )
    else:
        print("\n⚠️ 没有成功的测试数据，无法绘图。")

    print(f"\n✅ 测试完成。总耗时: {total_duration:.1f}秒")

# -------------------- 命令行入口 --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 吞吐量压测 + 专业报告生成")
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
