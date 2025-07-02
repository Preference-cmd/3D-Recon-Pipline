import argparse
from main import main as extract_main

def main():
    parser = argparse.ArgumentParser(description="Extract frames from video using config file.")
    parser.add_argument('--config', type=str, default='extract_config.yaml', help='Path to config YAML')
    args = parser.parse_args()
    # 直接调用main.py的main，利用其命令行和配置合并逻辑
    extract_main()

if __name__ == '__main__':
    main() 