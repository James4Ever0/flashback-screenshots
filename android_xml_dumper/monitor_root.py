#!/usr/bin/env python3
import subprocess
import xml.etree.ElementTree as ET
import re
import time
import sys
import os

def run_su_command(cmd):
    """以 root 权限执行命令，返回 (stdout, stderr)"""
    try:
        full_cmd = ["su", "-c"] + cmd.split()
        proc = subprocess.run(full_cmd, capture_output=True, text=True, timeout=10)
        return proc.stdout.strip(), proc.stderr.strip()
    except Exception as e:
        return "", str(e)


def get_display_and_lock_state():
    """参考 SWM 项目：通过 dumpsys power 的 mHolding 字段判断屏幕与锁屏状态。"""
    out, err = run_su_command("dumpsys power")
    if err:
        return "unknown"

    data = {}
    for line in out.split('\n'):
        if 'mHolding' in line and '=' in line:
            parts = line.strip().split()
            for part in parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    data[key.strip()] = value.strip()

    wake = data.get('mHoldingWakeLockSuspendBlocker')
    display = data.get('mHoldingDisplaySuspendBlocker')

    if wake == 'false' and display == 'false':
        return 'off_locked'
    if wake == 'true' and display == 'true':
        return 'on_unlocked'
    if wake == 'true' and display == 'false':
        return 'on_locked'
    if wake == 'false' and display == 'true':
        return 'off_unlocked'
    return 'unknown'


def is_screen_locked():
    """锁屏或状态未知时视为锁定，暂停采集。"""
    state = get_display_and_lock_state()
    return state in ('on_locked', 'off_locked', 'unknown')


def get_foreground_package():
    """通过 dumpsys 获取 display 0 上的前台应用包名（root）"""
    # 优先从 Display 0 区段提取 mCurrentFocus
    out, err = run_su_command("dumpsys window displays")
    if not err:
        in_display0 = False
        for line in out.split('\n'):
            if 'Display: mDisplayId=0' in line:
                in_display0 = True
            elif in_display0 and line.startswith('Display:'):
                break
            if in_display0 and 'mCurrentFocus' in line:
                match = re.search(r'([a-zA-Z0-9._]+)/', line)
                if match:
                    return match.group(1)

    # 兜底：直接用 dumpsys window 抓取当前焦点窗口
    out, err = run_su_command("dumpsys window")
    if not err:
        for line in out.split('\n'):
            if 'mCurrentFocus' in line or 'mFocusedWindow' in line:
                match = re.search(r'([a-zA-Z0-9._]+)/', line)
                if match:
                    return match.group(1)

    # 再兜底：从 activity 任务栈里找 resumed/top 活动
    out, err = run_su_command("dumpsys activity activities")
    if not err:
        for line in out.split('\n'):
            if 'mResumedActivity' in line or 'topResumedActivity' in line:
                match = re.search(r'([a-zA-Z0-9._]+)/', line)
                if match:
                    return match.group(1)

    if err:
        return f"[错误: {err}]"
    return "[未找到]"


def dump_ui_xml(tmp_xml):
    """执行 uiautomator dump，返回 (xml内容, 错误信息)。失败时自动重试最多3次。"""
    for attempt in range(1, 4):
        run_su_command(f"rm -f {tmp_xml}")
        out, err = run_su_command(f"uiautomator dump {tmp_xml}")
        if err and "Error" in err:
            if attempt == 3:
                return None, f"uiautomator 错误: {err}"
            time.sleep(0.5)
            continue

        xml_out, xml_err = run_su_command(f"cat {tmp_xml}")
        if xml_err:
            if attempt == 3:
                return None, f"读取 XML 错误: {xml_err}"
            time.sleep(0.5)
            continue

        # 校验 XML 是否为空或不完整
        if not xml_out or '<hierarchy' not in xml_out:
            if attempt == 3:
                return None, "dump 得到的 XML 为空或格式异常"
            time.sleep(0.5)
            continue

        return xml_out, None

    return None, "uiautomator dump 重试3次后仍失败"


def get_sample(max_retries=3):
    """同时获取前台应用包名与屏幕可见文字。XML dump 失败时整体重试，
    每次重试都重新获取包名；包名为空时不触发重试，但会用非空值更新上一次结果。
    """
    tmp_xml = "/data/local/tmp/ui_dump.xml"
    pkg = None
    texts = None

    for attempt in range(1, max_retries + 1):
        pkg = get_foreground_package()
        xml_out, err = dump_ui_xml(tmp_xml)
        if xml_out is not None:
            texts = _extract_texts_from_xml(xml_out)
            break
        if attempt < max_retries:
            time.sleep(0.5)

    run_su_command(f"rm -f {tmp_xml}")

    if texts is None:
        texts = [err or "uiautomator dump 失败"]

    return pkg, texts


def _extract_texts_from_xml(xml_out):
    """从 uiautomator XML 中提取可见文字。"""
    texts = []
    seen = set()
    # 优先用正则提取，兼容非标准 XML
    for attr in ('text', 'content-desc'):
        for match in re.finditer(rf'{attr}="([^"]*)"', xml_out):
            value = match.group(1).strip()
            if value and value not in seen:
                seen.add(value)
                texts.append(value)
    if not texts:
        # 正则未命中时回退到 ElementTree 解析
        root = ET.fromstring(xml_out)
        for elem in root.iter():
            text = elem.get('text', '').strip()
            if text and text not in seen:
                seen.add(text)
                texts.append(text)
            desc = elem.get('content-desc', '').strip()
            if desc and desc not in seen:
                seen.add(desc)
                texts.append(desc)
    return texts


def main():
    print("开始监控前台应用及屏幕文字 (每10秒，使用su权限)...")
    print("-" * 60)
    last_pkg = "[未找到]"
    while True:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # 屏幕锁定时跳过前台应用检测与 UI dump
        if is_screen_locked():
            print(f"[{timestamp}]")
            print("  屏幕已锁定，暂停采集")
            print("-" * 60)
            time.sleep(10)
            continue

        pkg, texts = get_sample()

        # 只有当前包名非空/有效时才更新上一次的包名记录
        if pkg and pkg not in ('[未找到]', '[错误]') and not pkg.startswith('[错误:'):
            last_pkg = pkg
        else:
            pkg = last_pkg

        print(f"[{timestamp}]")
        print(f"  前台应用包名: {pkg}")
        print(f"  屏幕可见文字 ({len(texts)} 条):")
        for i, text in enumerate(texts[:20], 1):
            print(f"    {i}. {text}")
        if len(texts) > 20:
            print(f"    ... 还有 {len(texts)-20} 条")
        print("-" * 60)
        time.sleep(10)


if __name__ == "__main__":
    try:
        # 检查是否有 root 权限
        test_out, test_err = run_su_command("id")
        if "uid=0" not in test_out:
            print("❌ 没有 root 权限，请确保已 root 并授予 Termux su 权限。", file=sys.stderr)
            sys.exit(1)
        main()
    except KeyboardInterrupt:
        print("\n✅ 监控已停止")
        sys.exit(0)
