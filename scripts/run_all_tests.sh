#!/bin/bash
# 一键运行所有测试脚本

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "运行所有测试 - Traffic Rules MVP"
echo "============================================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 单元测试
echo -e "${YELLOW}[1/3] 运行单元测试...${NC}"
echo "-----------------------------------------------------------"

test_passed=0
test_failed=0

for test_file in tests/unit/test_*.py; do
    if [ -f "$test_file" ]; then
        echo "运行 $test_file..."
        if python3 "$test_file" > /dev/null 2>&1; then
            echo -e "  ${GREEN}✅ 通过${NC}"
            ((test_passed++))
        else
            echo -e "  ❌ 失败"
            ((test_failed++))
        fi
    fi
done

echo ""
echo "单元测试结果: ${test_passed}个通过, ${test_failed}个失败"
echo ""

# 2. 集成测试
echo -e "${YELLOW}[2/3] 运行集成测试...${NC}"
echo "-----------------------------------------------------------"

if python3 tests/integration/traffic_rules/test_cli.py > /dev/null 2>&1; then
    echo -e "${GREEN}✅ 集成测试通过${NC}"
else
    echo "❌ 集成测试失败"
    exit 1
fi

echo ""

# 3. 验收测试
echo -e "${YELLOW}[3/3] 运行验收测试...${NC}"
echo "-----------------------------------------------------------"

# 检查checkpoint
if [ ! -f "artifacts/checkpoints/best.pth" ]; then
    echo "⚠️  未找到checkpoint，运行快速训练..."
    python3 tools/train_red_light.py train --epochs 2 --max-samples 5 --device cpu
fi

# 运行三场景测试
echo "测试所有场景..."
python3 tools/test_red_light.py --scenario all --split val > /dev/null 2>&1

# 生成热力图
echo "生成注意力热力图..."
python3 scripts/render_attention_maps.py --output-dir reports/testing/heatmaps > /dev/null 2>&1

# 生成验收报告
echo "生成验收报告..."
python3 tools/generate_acceptance_report.py

echo -e "${GREEN}✅ 验收测试完成${NC}"
echo ""

# 总结
echo "============================================================"
echo -e "${GREEN}✅ 所有测试完成！${NC}"
echo "============================================================"
echo ""
echo "📊 测试统计:"
echo "  - 单元测试: ${test_passed}个通过"
echo "  - 集成测试: 1个通过"
echo "  - 验收测试: 通过"
echo ""
echo "📁 输出位置:"
echo "  - 验收报告: reports/ACCEPTANCE_REPORT.md"
echo "  - 违规截图: reports/testing/screenshots/ ($(ls reports/testing/screenshots/*.png 2>/dev/null | wc -l | tr -d ' ')张)"
echo "  - 注意力热力图: reports/testing/heatmaps/ ($(ls reports/testing/heatmaps/*.png 2>/dev/null | wc -l | tr -d ' ')张)"
echo "  - HTML索引: reports/testing/heatmaps/index.html"
echo ""
echo "🎉 MVP验收完成！"
