"""
MCP 服务器配置模块
=================================

此模块处理 MCP 服务器的所有配置，
包括路径、Unstructured 配置和系统参数。
"""

import os
from dotenv import load_dotenv, dotenv_values
from typing import Dict, Any
from pathlib import Path

# 加载环境变量
load_dotenv()

class Config:
    """
    MCP 服务器的集中配置类。
    """
    
    # 项目根目录
    ROOT_DIR = Path(__file__).resolve().parents[2]
    DOTENV_PATH = ROOT_DIR / '.env'
    
    # 服务器配置
    SERVER_NAME = "ragmcp"
    SERVER_VERSION = "1.0.0"
    
    # Web配置
    WEB_HOST = "0.0.0.0"
    WEB_PORT = 5000
    WEB_DEBUG = True
    UPLOAD_FOLDER = "./data/rag/uploads"
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
    
    # 数据路径
    CONVERTED_DOCS_DIR = "./data/rag/documents"
    VECTOR_STORE_DIR = "./data/rag/vector_store"
    EMBEDDING_CACHE_DIR = "./data/rag/embedding_cache"
    
    # 模型配置
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    DEVICE = "cpu"
    
    # 分块配置
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    
    # 缓存配置
    MAX_CACHE_SIZE = 1000
    
    # 针对不同文档类型的优化配置
    UNSTRUCTURED_CONFIGS = {
        # Office 文档
        '.pdf': {
            'strategy': 'hi_res',
            'include_metadata': True,
            'include_page_breaks': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        '.docx': {
            'strategy': 'fast',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        '.doc': {
            'strategy': 'fast',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        '.pptx': {
            'strategy': 'fast',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        '.ppt': {
            'strategy': 'fast',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        '.xlsx': {
            'strategy': 'fast',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        '.xls': {
            'strategy': 'fast',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        '.rtf': {
            'strategy': 'fast',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        
        # OpenDocument 文档
        '.odt': {
            'strategy': 'fast',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        '.odp': {
            'strategy': 'fast',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        '.ods': {
            'strategy': 'fast',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        
        # Web 和标记格式
        '.html': {
            'strategy': 'fast',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        '.htm': {
            'strategy': 'fast',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        '.xml': {
            'strategy': 'fast',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        '.md': {
            'strategy': 'fast',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        
        # 纯文本格式
        '.txt': {
            'strategy': 'fast',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        '.csv': {
            'strategy': 'fast',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        '.tsv': {
            'strategy': 'fast',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        
        # 数据格式
        '.json': {
            'strategy': 'fast',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        '.yaml': {
            'strategy': 'fast',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        '.yml': {
            'strategy': 'fast',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        
        # 图像（需要 OCR）
        '.png': {
            'strategy': 'hi_res',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        '.jpg': {
            'strategy': 'hi_res',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        '.jpeg': {
            'strategy': 'hi_res',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        '.tiff': {
            'strategy': 'hi_res',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        '.bmp': {
            'strategy': 'hi_res',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        
        # 电子邮件
        '.eml': {
            'strategy': 'fast',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        },
        '.msg': {
            'strategy': 'fast',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        }
    }
    
    @classmethod
    def get_unstructured_config(cls, file_extension: str) -> Dict[str, Any]:
        """
        获取特定文件类型的 Unstructured 配置。
        
        Args:
            file_extension: 文件扩展名（例如：'.pdf'）
            
        Returns:
            文件类型的 Unstructured 配置
        """
        return cls.UNSTRUCTURED_CONFIGS.get(file_extension.lower(), {
            'strategy': 'fast',
            'include_metadata': True,
            'max_partition': 2000,
            'new_after_n_chars': 1500
        })
    
    @classmethod
    def ensure_directories(cls):
        """
        确保所有必要的目录都存在。
        """
        directories = [
            cls.CONVERTED_DOCS_DIR,
            cls.VECTOR_STORE_DIR,
            cls.EMBEDDING_CACHE_DIR
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    @classmethod
    def get_env_var(cls, key: str, default: str = None) -> str:
        """
        获取环境变量并提供默认值。
        
        Args:
            key: 环境变量名称
            default: 如果不存在的默认值
            
        Returns:
            环境变量的值或默认值
        """
        return os.getenv(key, default)
    
    @classmethod
    def load_env_cache(cls) -> Dict[str, str]:
        """
        加载.env文件的缓存。
        
        Returns:
            环境变量字典
        """
        if cls.DOTENV_PATH.exists():
            return dict(dotenv_values(cls.DOTENV_PATH))
        return {}
    
    @classmethod
    def save_env_vars(cls, env_vars: Dict[str, str]) -> bool:
        """
        保存环境变量到.env文件和系统环境。
        
        Args:
            env_vars: 要保存的环境变量字典
            
        Returns:
            保存是否成功
        """
        try:
            # 更新系统环境变量
            for key, value in env_vars.items():
                if value:
                    os.environ[key] = value
            
            # 加载现有.env内容
            existing_env = cls.load_env_cache()
            existing_env.update(env_vars)
            
            # 写回.env文件
            with open(cls.DOTENV_PATH, 'w', encoding='utf-8') as f:
                for key, value in existing_env.items():
                    if value:  # 只保存非空值
                        f.write(f"{key}={value}\n")
            
            return True
        except Exception as e:
            print(f"保存环境变量失败: {e}")
            return False
    
    @classmethod
    def get_web_env_vars(cls) -> Dict[str, str]:
        """
        获取Web界面使用的环境变量。
        
        Returns:
            Web界面环境变量字典
        """
        return {
            'OPENAI_API_KEY': cls.get_env_var('OPENAI_API_KEY', ''),
            'OPENAI_API_BASE': cls.get_env_var('OPENAI_API_BASE', 'https://ark.cn-beijing.volces.com/api/v3'),
            'OPENAI_MODEL': cls.get_env_var('OPENAI_MODEL', 'doubao-1-5-pro-32k-250115'),
            'OPENAI_EMBEDDING_MODEL': cls.get_env_var('OPENAI_EMBEDDING_MODEL', 'doubao-embedding-text-240715'),
            'OPENAI_TEMPERATURE': cls.get_env_var('OPENAI_TEMPERATURE', '0')
        }
    
    @classmethod
    def check_required_env_vars(cls) -> Dict[str, Any]:
        """
        检查必要的环境变量是否已配置。
        
        Returns:
            检查结果字典
        """
        required_vars = ['OPENAI_API_KEY']
        missing = []
        
        for var in required_vars:
            if not cls.get_env_var(var):
                missing.append(var)
        
        return {
            'configured': len(missing) == 0,
            'missing': missing,
            'has_api_base': bool(cls.get_env_var('OPENAI_API_BASE')),
            'model': cls.get_env_var('OPENAI_MODEL', 'doubao-1-5-pro-32k-250115'),
            'embedding_model': cls.get_env_var('OPENAI_EMBEDDING_MODEL', 'doubao-embedding-text-240715'),
            'temperature': cls.get_env_var('OPENAI_TEMPERATURE', '0')
        }