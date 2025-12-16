import os
from pathlib import Path

# 基础路径配置
class Paths:


    # 外部数据集路径（可根据环境变量覆盖）
    EXTERNAL_DATASETS = Path(os.getenv("DATASET_PATH", "/Users/shiyifan/Documents/dataset"))
    
    # 具体数据集
    BDD100K = EXTERNAL_DATASETS / "BDD100K"

    
   