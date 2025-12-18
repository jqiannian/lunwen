"""
MemoryBank 单元测试
"""

import sys
from pathlib import Path
import tempfile

import torch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.traffic_rules.memory.memory_bank import MemoryBank


class TestMemoryBank:
    """测试MemoryBank类"""
    
    def test_initialization(self):
        """测试初始化"""
        mb = MemoryBank(size=128, embedding_dim=64)
        
        assert mb.size == 128
        assert mb.embedding_dim == 64
        assert mb.storage.shape == (128, 64)
        
        print("✅ MemoryBank初始化成功")
    
    def test_query(self):
        """测试查询功能"""
        mb = MemoryBank(size=10, embedding_dim=8)
        
        # 初始化一些记忆
        mb.storage = torch.randn(10, 8)
        
        # 查询
        query_emb = torch.randn(3, 8)
        result = mb.query(query_emb)
        
        assert result.shape == (3, 8), f"查询结果维度错误: {result.shape}"
        
        print(f"✅ 查询功能正常: {result.shape}")
    
    def test_save_and_load(self):
        """测试保存和加载"""
        mb1 = MemoryBank(size=10, embedding_dim=8)
        mb1.storage = torch.randn(10, 8)
        
        # 保存
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "memory.pth"
            mb1.save(save_path)
            
            assert save_path.exists(), "保存文件应存在"
            
            # 加载
            mb2 = MemoryBank(size=10, embedding_dim=8)
            mb2.load(save_path)
            
            # 验证一致性
            assert torch.allclose(mb1.storage, mb2.storage), "加载后数据应一致"
        
        print("✅ 保存和加载功能正常")
    
    def test_update_ema(self):
        """测试EMA更新（简化版）"""
        mb = MemoryBank(size=5, embedding_dim=4)
        original_storage = torch.randn(5, 4)
        mb.storage = original_storage.clone()
        
        # 更新
        new_emb = torch.randn(2, 4)
        mb.update(new_emb, ema_decay=0.9)
        
        # 验证更新后不完全相同（已被更新）
        assert not torch.allclose(mb.storage, original_storage), "应该已更新"
        
        print("✅ EMA更新功能正常")
    
    def test_reset(self):
        """测试重置"""
        mb = MemoryBank(size=10, embedding_dim=8)
        mb.storage = torch.randn(10, 8)
        
        mb.reset()
        
        assert torch.allclose(mb.storage, torch.zeros(10, 8)), "重置后应为全0"
        
        print("✅ 重置功能正常")


if __name__ == "__main__":
    test = TestMemoryBank()
    
    test.test_initialization()
    test.test_query()
    test.test_save_and_load()
    test.test_update_ema()
    test.test_reset()
    
    print("\n" + "="*60)
    print("✅ 所有MemoryBank测试通过！")
    print("="*60)
