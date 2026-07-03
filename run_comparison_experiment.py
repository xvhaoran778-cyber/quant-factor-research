#!/usr/bin/env python3
"""
因子对比实验运行脚本

运行 6 个实验对比：
1. Baseline: Alpha191 (α132 + ret60 + size)
2. Baseline + Agent
3. New Factors: 15 个新因子
4. New Factors + Agent
5. Hybrid: Alpha191 + 新因子
6. Hybrid + Agent
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
backend_root = project_root / "backend"
sys.path.insert(0, str(backend_root))

from app.services.comparison_experiment import run_comparison_experiment


def main():
    """主函数"""
    print("="*60)
    print("因子对比实验")
    print("="*60)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 配置参数
    data_dir = "/Volumes/xhrrrrr_macmini副盘/quantlab/market"
    
    # 实验区间
    start_date = "2020-01-01"
    end_date = "2024-12-31"
    
    # 选择的新因子数量
    top_n_factors = 5
    
    # 输出文件
    output_dir = project_root / "reports"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"comparison_experiment_{timestamp}.txt"
    
    print(f"数据目录: {data_dir}")
    print(f"实验区间: {start_date} 至 {end_date}")
    print(f"选择因子数: {top_n_factors}")
    print(f"输出文件: {output_file}")
    print()
    
    # 运行实验
    try:
        experiment, comparison_df, improvement_df = run_comparison_experiment(
            data_dir=data_dir,
            start_date=start_date,
            end_date=end_date,
            top_n_factors=top_n_factors,
            output_file=str(output_file)
        )
        
        print("\n" + "="*60)
        print("实验完成")
        print("="*60)
        
        # 保存详细结果
        results_file = output_dir / f"comparison_results_{timestamp}.csv"
        comparison_df.to_csv(results_file, index=False, encoding='utf-8-sig')
        print(f"详细结果已保存到: {results_file}")
        
        # 如果有改善数据，也保存
        if not improvement_df.empty:
            improvement_file = output_dir / f"agent_improvement_{timestamp}.csv"
            improvement_df.to_csv(improvement_file, index=False, encoding='utf-8-sig')
            print(f"Agent 改善数据已保存到: {improvement_file}")
        
        print("\n实验完成！")
        
    except Exception as e:
        print(f"\n实验运行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
