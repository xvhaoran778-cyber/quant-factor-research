# 因子计算修复总结

## 问题描述

在运行因子对比实验时，所有因子的 IC 值都是 NaN，导致无法评估因子质量。

## 根本原因

在 `backend/app/services/new_factors.py` 的 `compute_all_factors` 函数中，第 597 行使用了以下代码：

```python
main_data = kwargs.get('returns') or kwargs.get('close') or kwargs.get('prices')
```

当 `kwargs.get('returns')` 返回一个 pandas Series 时，Python 会尝试评估其真值（用于 `or` 操作符），这会触发以下错误：

```
ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
```

## 修复方案

将 `or` 操作符替换为显式的键检查：

```python
main_data = None
for key in ['returns', 'close', 'prices']:
    if key in kwargs:
        main_data = kwargs[key]
        break
```

这样可以避免对 Series 对象进行布尔评估。

## 其他修复

1. **数据充足性检查**：添加了检查，确保有足够的数据点（至少 60 个）才计算因子
2. **边界情况处理**：修复了 `return_skewness` 和 `return_kurtosis` 因子在数据不足时的处理
3. **返回值类型**：确保因子函数返回标量值而非 Series

## 验证结果

修复后，因子计算可以正确生成非 NaN 值：

- **因子值**：2260 个非 NaN 值（之前是 0）
- **IC 值**：119 个有效 IC 值（之前是 0）
- **IC 统计**：
  - 均值：-0.12
  - 标准差：0.34
  - 范围：-0.83 到 0.79

## 当前状态

- ✅ 因子计算框架正常工作
- ✅ IC 分析框架正常工作
- ✅ 对比实验框架正常工作
- ⚠️ 完整实验需要较长时间（5000+ 股票 × 5 年数据）

## 下一步建议

1. 运行完整对比实验（可能需要数小时）
2. 分析哪些因子具有预测能力（IC 绝对值 > 0.03）
3. 筛选最佳因子组合
4. 构建增强版 V8 策略

## 相关文件

- `backend/app/services/new_factors.py` - 因子计算
- `backend/app/services/factor_ic_analysis.py` - IC 分析
- `backend/app/services/comparison_experiment.py` - 对比实验
- `debug_factor_ic.py` - 调试脚本
- `test_single_factor.py` - 单因子测试脚本
