import json
import curses

# 只保留的指标字段
METRIC_FIELDS = [
    "generated_answers_0_agg_runtime_adjusted",
    "fastest_generated_agg_runtime",
    "fastest_generated_runtime_over_all_submissions",
    "fastest_generated_speedup_over_all_submissions",
    "fastest_generated_correctness_over_all_submissions",
    "agg_runtime_best@1",
    "accuracy_best@1",
    "is_correct_best@1",
    "speedup_best@1",
    "speedup_of_fastest_generated_of_all_submissions",
    "speedup_tgt_over_src",
    "problem_id"
]

def read_jsonl(file_path):
    """读取 jsonl 文件并提取 src_code, tgt_code, generated_answers, analysis 和指定指标字段"""
    items = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                src = obj.get("src_code", "")
                tgt = obj.get("tgt_code", "")
                gen = obj.get("generated_answers", "")
                ana = obj.get("analysis", "")

                # 处理 generated_answers 类型
                if isinstance(gen, list):
                    gen = "\n".join(str(x) for x in gen)
                elif not isinstance(gen, str):
                    gen = str(gen)
                if not gen.strip():
                    gen = "<EMPTY>"

                # 处理 analysis 类型（格式化为缩进 JSON）
                if isinstance(ana, (dict, list)):
                    ana = json.dumps(ana, indent=2, ensure_ascii=False)
                elif not isinstance(ana, str):
                    ana = str(ana)

                # 只提取指定的指标
                metrics = {k: obj.get(k, None) for k in METRIC_FIELDS if k in obj}

                items.append((src, tgt, gen, metrics, ana))
            except json.JSONDecodeError:
                pass
    return items


def wrap_text(text, width):
    """按指定宽度换行"""
    lines = []
    for line in str(text).splitlines():
        while len(line) > width:
            lines.append(line[:width])
            line = line[width:]
        lines.append(line)
    return lines

def render_item(stdscr, item, scroll, index, total):
    stdscr.clear()
    max_y, max_x = stdscr.getmaxyx()

    src, tgt, gen, metrics, ana = item

    # 上面两列：SRC 和 TGT
    col_width = max_x // 2 - 1
    src_lines = wrap_text(src, col_width)
    tgt_lines = wrap_text(tgt, col_width)

    lines = []
    # 表头
    lines.append("SRC_CODE".ljust(col_width) + "  " + "TGT_CODE")

    # 并排拼接 src/tgt
    max_code_lines = max(len(src_lines), len(tgt_lines))
    for i in range(max_code_lines):
        left = src_lines[i] if i < len(src_lines) else ""
        right = tgt_lines[i] if i < len(tgt_lines) else ""
        lines.append(left.ljust(col_width) + "  " + right)

    # generated_answers
    lines.append("")
    lines.append("GENERATED_ANSWERS:")
    lines.extend(wrap_text(gen, max_x - 1))

    # metrics（固定 11 个字段）
    if metrics:
        lines.append("")
        lines.append("METRICS:")
        for field in METRIC_FIELDS:
            if field in metrics:
                value = metrics[field]
                lines.extend(wrap_text(f"{field}: {value}", max_x - 4))

    # analysis
    lines.append("")
    lines.append("ANALYSIS:")
    lines.extend(wrap_text(ana, max_x - 1))

    # --- 滚动显示 ---
    visible_height = max_y - 2
    if scroll < 0:
        scroll = 0
    if scroll > max(0, len(lines) - visible_height):
        scroll = max(0, len(lines) - visible_height)

    for i, line in enumerate(lines[scroll:scroll+visible_height]):
        stdscr.addstr(i, 0, line[:max_x-1])

    footer = f"[←上一条 →下一条 ↑↓滚动 q退出] 条目 {index+1}/{total}"
    stdscr.addstr(max_y - 1, 0, footer[:max_x-1])
    stdscr.refresh()
    return scroll

def render_item(stdscr, item, scroll, index, total, view_mode):
    stdscr.clear()
    max_y, max_x = stdscr.getmaxyx()
    src, tgt, gen, metrics, ana = item

    lines = []
    if view_mode == "split":  # 并排
        col_width = max_x // 2 - 1
        src_lines = wrap_text(src, col_width)
        tgt_lines = wrap_text(tgt, col_width)

        lines.append("SRC_CODE".ljust(col_width) + "  " + "TGT_CODE")
        max_code_lines = max(len(src_lines), len(tgt_lines))
        for i in range(max_code_lines):
            left = src_lines[i] if i < len(src_lines) else ""
            right = tgt_lines[i] if i < len(tgt_lines) else ""
            lines.append(left.ljust(col_width) + "  " + right)
    elif view_mode == "src":
        lines.append("SRC_CODE")
        lines.extend(wrap_text(src, max_x - 1))
    elif view_mode == "tgt":
        lines.append("TGT_CODE")
        lines.extend(wrap_text(tgt, max_x - 1))

    # generated_answers
    lines.append("")
    lines.append("GENERATED_ANSWERS:")
    lines.extend(wrap_text(gen, max_x - 1))

    # metrics
    if metrics:
        lines.append("")
        lines.append("METRICS:")
        for field in METRIC_FIELDS:
            if field in metrics:
                value = metrics[field]
                lines.extend(wrap_text(f"{field}: {value}", max_x - 4))

    # analysis
    lines.append("")
    lines.append("ANALYSIS:")
    lines.extend(wrap_text(ana, max_x - 1))

    # --- 滚动显示 ---
    visible_height = max_y - 2
    if scroll < 0:
        scroll = 0
    if scroll > max(0, len(lines) - visible_height):
        scroll = max(0, len(lines) - visible_height)

    for i, line in enumerate(lines[scroll:scroll+visible_height]):
        stdscr.addstr(i, 0, line[:max_x-1])

    footer = f"[←上一条 →下一条 ↑↓滚动 Tab切换视图 q退出] 条目 {index+1}/{total} 视图:{view_mode}"
    stdscr.addstr(max_y - 1, 0, footer[:max_x-1])
    stdscr.refresh()
    return scroll

def main(stdscr, file_path="/data/btguan/pie/test_sep7_2/addtl_stats.jsonl"):
    curses.curs_set(0)
    stdscr.keypad(True)

    items = read_jsonl(file_path)
    if not items:
        stdscr.addstr(0, 0, "未找到任何字段")
        stdscr.refresh()
        stdscr.getch()
        return

    index = 0
    scroll = 0
    view_modes = ["split", "src", "tgt"]
    view_idx = 0
    while True:
        scroll = render_item(stdscr, items[index], scroll, index, len(items), view_modes[view_idx])
        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            break
        elif key == curses.KEY_RIGHT:
            index = (index + 1) % len(items)
            scroll = 0
        elif key == curses.KEY_LEFT:
            index = (index - 1) % len(items)
            scroll = 0
        elif key == curses.KEY_DOWN:
            scroll += 1
        elif key == curses.KEY_UP:
            scroll -= 1
        elif key == 9:  # Tab 键
            view_idx = (view_idx + 1) % len(view_modes)
            scroll = 0

if __name__ == "__main__":
    curses.wrapper(main)