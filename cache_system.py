import os
import shutil
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import gc

class CacheManager:
    """
    統一管理快取檔案的類別，提供層次化的資料夾結構和標準化的存取方法
    """
    
    # 快取根目錄
    BASE_CACHE_DIR = 'D:/data_cache'
    
    # 快取分類定義
    STRUCTURE = {
        'raw_data': {               # 原始資料快取
            'stock_price': {},      # 股價數據
            'margin_data': {},      # 融資融券數據
            'foreign_trade': {},    # 外資交易數據
            'broker_data': {},      # 券商分點數據
            'stock_info': {}        # 股票基本資料
        },
        'features': {               # 特徵工程結果快取
            'price_features': {},   # 價格特徵
            'volume_features': {},  # 成交量特徵
            'margin_features': {},  # 融資融券特徵
            'foreign_features': {}, # 外資特徵
            'broker_features': {},  # 券商分點特徵
            'combined_features': {} # 合併特徵
        },
        'models': {                 # 模型相關快取
            'xgboost': {},          # XGBoost 模型
            'lightgbm': {},         # LightGBM 模型
            'tabnet': {},           # TabNet 模型
            'scalers': {}           # 標準化參數
        },
        'predictions': {            # 預測結果快取
            'train': {},            # 訓練集預測
            'test': {}              # 測試集預測
        },
        'temp': {}                  # 臨時檔案
    }
    
    @classmethod
    def initialize(cls):
        """創建快取目錄結構"""
        # 建立基礎目錄
        if not os.path.exists(cls.BASE_CACHE_DIR):
            os.makedirs(cls.BASE_CACHE_DIR)
        
        # 建立子目錄結構
        cls._create_folder_structure(cls.STRUCTURE, cls.BASE_CACHE_DIR)
        
        print(f"快取系統初始化完成，根目錄：{cls.BASE_CACHE_DIR}")
        return cls.BASE_CACHE_DIR
    
    @classmethod
    def _create_folder_structure(cls, structure, parent_path):
        """遞迴建立資料夾結構"""
        for folder, subfolders in structure.items():
            folder_path = os.path.join(parent_path, folder)
            # 建立目錄
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            # 遞迴建立子目錄
            if subfolders:
                cls._create_folder_structure(subfolders, folder_path)
    
    @classmethod
    def get_path(cls, category, subcategory=None):
        """
        取得特定類別的快取目錄路徑
        
        Parameters:
        -----------
        category : str
            主要分類 ('raw_data', 'features', 'models', 'predictions')
        subcategory : str, optional
            次要分類 (如 'stock_price', 'broker_features')
            
        Returns:
        --------
        str
            完整的快取目錄路徑
        """
        if category not in cls.STRUCTURE:
            print(f"警告：未知的快取類別 '{category}'，使用 'temp' 目錄")
            category = 'temp'
        
        path = os.path.join(cls.BASE_CACHE_DIR, category)
        
        if subcategory:
            # 檢查子類別是否存在
            if category in cls.STRUCTURE and subcategory in cls.STRUCTURE[category]:
                path = os.path.join(path, subcategory)
            else:
                print(f"警告：未知的快取子類別 '{subcategory}'，使用 '{category}' 根目錄")
        
        # 確保目錄存在
        os.makedirs(path, exist_ok=True)
        return path
    
    @classmethod
    def get_file_path(cls, category, subcategory, filename):
        """
        取得完整的快取檔案路徑
        
        Parameters:
        -----------
        category : str
            主要分類
        subcategory : str
            次要分類
        filename : str
            檔案名稱
            
        Returns:
        --------
        str
            完整的快取檔案路徑
        """
        path = cls.get_path(category, subcategory)
        return os.path.join(path, filename)
    
    @classmethod
    def exists(cls, category, subcategory, filename):
        """檢查快取檔案是否存在"""
        file_path = cls.get_file_path(category, subcategory, filename)
        return os.path.exists(file_path)
    
    @classmethod
    def save(cls, data, category, subcategory, filename):
        """
        儲存資料到快取
        
        Parameters:
        -----------
        data : object
            要儲存的資料
        category : str
            主要分類
        subcategory : str
            次要分類
        filename : str
            檔案名稱
            
        Returns:
        --------
        str
            儲存的檔案路徑
        """
        file_path = cls.get_file_path(category, subcategory, filename)
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"已儲存快取: {file_path}")
            return file_path
        except Exception as e:
            print(f"儲存快取失敗 ({file_path}): {e}")
            return None
    
    @classmethod
    def load(cls, category, subcategory, filename):
        """
        從快取載入資料
        
        Parameters:
        -----------
        category : str
            主要分類
        subcategory : str
            次要分類
        filename : str
            檔案名稱
            
        Returns:
        --------
        object or None
            載入的資料，如果失敗則返回 None
        """
        file_path = cls.get_file_path(category, subcategory, filename)
        
        if not os.path.exists(file_path):
            print(f"快取檔案不存在: {file_path}")
            return None
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            print(f"已讀取快取: {file_path}")
            return data
        except Exception as e:
            print(f"讀取快取失敗 ({file_path}): {e}")
            return None
    
    @classmethod
    def list_files(cls, category=None, subcategory=None, pattern="*"):
        """
        列出特定類別的快取檔案
        
        Parameters:
        -----------
        category : str, optional
            主要分類，如果不指定則列出所有類別
        subcategory : str, optional
            次要分類，如果不指定則列出該類別下所有子類別
        pattern : str, default="*"
            檔案名稱模式，支援萬用字元
            
        Returns:
        --------
        list of str
            符合條件的檔案路徑列表
        """
        if category:
            path = cls.get_path(category, subcategory)
            return glob.glob(os.path.join(path, pattern))
        else:
            # 列出所有快取檔案
            all_files = []
            for root, _, files in os.walk(cls.BASE_CACHE_DIR):
                for file in files:
                    if fnmatch.fnmatch(file, pattern):
                        all_files.append(os.path.join(root, file))
            return all_files
    
    @classmethod
    def clear(cls, category=None, subcategory=None, pattern="*", confirm=True):
        """
        清除特定類別的快取檔案
        
        Parameters:
        -----------
        category : str, optional
            主要分類，如果不指定則清除所有類別
        subcategory : str, optional
            次要分類，如果不指定則清除該類別下所有子類別
        pattern : str, default="*"
            檔案名稱模式，支援萬用字元
        confirm : bool, default=True
            是否要求確認
            
        Returns:
        --------
        int
            清除的檔案數量
        """
        # 取得要清除的檔案
        files = cls.list_files(category, subcategory, pattern)
        
        if not files:
            print("沒有符合條件的快取檔案需要清除")
            return 0
        
        # 計算檔案總大小
        total_size = sum(os.path.getsize(file) for file in files)
        size_str = cls._format_size(total_size)
        
        # 顯示清除資訊
        if confirm:
            print(f"將清除 {len(files)} 個快取檔案，總大小 {size_str}")
            
            # 顯示部分檔案
            max_show = min(10, len(files))
            for i in range(max_show):
                print(f"  - {os.path.basename(files[i])}")
            
            if len(files) > max_show:
                print(f"  ... 以及其他 {len(files) - max_show} 個檔案")
            
            # 確認
            confirm_input = input("確定要清除這些快取檔案嗎？(y/n): ")
            if confirm_input.lower() != 'y':
                print("取消清除")
                return 0
        
        # 執行清除
        deleted_count = 0
        for file in files:
            try:
                os.remove(file)
                deleted_count += 1
            except Exception as e:
                print(f"無法刪除 {file}: {e}")
        
        print(f"已清除 {deleted_count}/{len(files)} 個快取檔案，釋放空間 {size_str}")
        return deleted_count
    
    @classmethod
    def _format_size(cls, size_bytes):
        """格式化檔案大小"""
        if size_bytes >= 1e9:
            return f"{size_bytes/1e9:.2f} GB"
        elif size_bytes >= 1e6:
            return f"{size_bytes/1e6:.2f} MB"
        elif size_bytes >= 1e3:
            return f"{size_bytes/1e3:.2f} KB"
        else:
            return f"{size_bytes} bytes"
    
    @classmethod
    def migrate_old_cache(cls, old_cache_dir=None, confirm=True):
        """
        將舊的快取檔案遷移到新的結構中
        
        Parameters:
        -----------
        old_cache_dir : str, optional
            舊快取目錄，默認為基本快取目錄
        confirm : bool, default=True
            是否要求確認
            
        Returns:
        --------
        dict
            遷移統計結果
        """
        if old_cache_dir is None:
            old_cache_dir = cls.BASE_CACHE_DIR
        
        # 確保新結構已初始化
        cls.initialize()
        
        # 統計物件
        stats = {
            'found': 0,
            'migrated': 0,
            'skipped': 0,
            'failed': 0,
            'total_size': 0
        }
        
        # 掃描舊的快取檔案
        old_files = []
        for root, _, files in os.walk(old_cache_dir):
            # 如果已經在分類目錄下，則跳過
            rel_path = os.path.relpath(root, old_cache_dir)
            if rel_path != '.' and rel_path in cls.STRUCTURE:
                continue
                
            for file in files:
                # 排除非pickle檔案
                if not file.endswith('.pkl'):
                    continue
                    
                file_path = os.path.join(root, file)
                stats['found'] += 1
                stats['total_size'] += os.path.getsize(file_path)
                
                # 猜測檔案類型
                category, subcategory = cls._guess_file_category(file)
                old_files.append((file_path, file, category, subcategory))
        
        if not old_files:
            print("未找到需要遷移的舊快取檔案")
            return stats
        
        # 顯示遷移計劃
        print(f"找到 {stats['found']} 個檔案需要遷移，總大小 {cls._format_size(stats['total_size'])}")
        
        if confirm:
            print("\n遷移計劃:")
            max_show = min(15, len(old_files))
            for i in range(max_show):
                print(f"  {old_files[i][1]} -> {old_files[i][2]}/{old_files[i][3]}/")
            
            if len(old_files) > max_show:
                print(f"  ... 以及其他 {len(old_files) - max_show} 個檔案")
            
            # 確認
            confirm_input = input("確定要執行遷移嗎？(y/n): ")
            if confirm_input.lower() != 'y':
                print("取消遷移")
                return stats
        
        # 執行遷移
        for old_path, filename, category, subcategory in old_files:
            if category == 'unknown':
                stats['skipped'] += 1
                continue
                
            # 建立目標路徑
            target_path = cls.get_file_path(category, subcategory, filename)
            
            # 如果源和目標相同，則跳過
            if os.path.abspath(old_path) == os.path.abspath(target_path):
                stats['skipped'] += 1
                continue
            
            # 如果目標已存在，則跳過
            if os.path.exists(target_path):
                print(f"目標已存在，跳過: {target_path}")
                stats['skipped'] += 1
                continue
            
            try:
                # 先嘗試複製，再刪除源文件
                shutil.copy2(old_path, target_path)
                if os.path.exists(target_path):
                    os.remove(old_path)
                
                stats['migrated'] += 1
                print(f"已遷移: {old_path} -> {target_path}")
            except Exception as e:
                stats['failed'] += 1
                print(f"遷移失敗: {old_path} -> {target_path}, 錯誤: {e}")
        
        # 顯示遷移結果
        print(f"\n遷移完成:")
        print(f"  發現: {stats['found']} 個檔案")
        print(f"  成功遷移: {stats['migrated']} 個檔案")
        print(f"  跳過: {stats['skipped']} 個檔案")
        print(f"  失敗: {stats['failed']} 個檔案")
        
        return stats
    
    @classmethod
    def _guess_file_category(cls, filename):
        """根據檔案名稱猜測其類別和子類別"""
        # 先處理原始數據
        if 'stock_data' in filename:
            return 'raw_data', 'stock_price'
        elif 'margin_data' in filename:
            return 'raw_data', 'margin_data'
        elif 'foreign_trade' in filename:
            return 'raw_data', 'foreign_trade'
        elif 'broker_data' in filename:
            return 'raw_data', 'broker_data'
        elif 'stock_info' in filename:
            return 'raw_data', 'stock_info'
        
        # 處理特徵
        elif 'price_features' in filename or 'p_' in filename:
            return 'features', 'price_features'
        elif 'volume_features' in filename or 'vol_' in filename:
            return 'features', 'volume_features'
        elif 'margin_features' in filename or 'm_' in filename or 's_' in filename:
            return 'features', 'margin_features'
        elif 'foreign_features' in filename or 'f_' in filename:
            return 'features', 'foreign_features'
        elif 'broker_features' in filename:
            return 'features', 'broker_features'
        elif 'final_features' in filename or 'combined_features' in filename:
            return 'features', 'combined_features'
        
        # 處理模型和其他檔案
        elif 'xgboost' in filename or 'xgb' in filename:
            return 'models', 'xgboost'
        elif 'lightgbm' in filename or 'lgb' in filename:
            return 'models', 'lightgbm'
        elif 'tabnet' in filename:
            return 'models', 'tabnet'
        elif 'scaler' in filename:
            return 'models', 'scalers'
        elif 'train_predict' in filename or 'train_pred' in filename:
            return 'predictions', 'train'
        elif 'test_predict' in filename or 'test_pred' in filename:
            return 'predictions', 'test'
        
        # 無法判斷，放入temp目錄
        return 'temp', None
    
    @classmethod
    def generate_report(cls):
        """生成快取使用報告"""
        report = []
        total_files = 0
        total_size = 0
        
        # 分析所有類別
        for category in cls.STRUCTURE:
            cat_files = 0
            cat_size = 0
            
            # 檢查這個類別下所有子類別
            subcat_info = []
            for subcategory in cls.STRUCTURE[category]:
                path = cls.get_path(category, subcategory)
                
                # 計算檔案數量和大小
                files = glob.glob(os.path.join(path, "*"))
                num_files = len(files)
                size = sum(os.path.getsize(f) for f in files)
                
                cat_files += num_files
                cat_size += size
                
                # 紀錄子類別資訊
                if num_files > 0:
                    subcat_info.append({
                        'subcategory': subcategory,
                        'files': num_files,
                        'size': size,
                        'size_str': cls._format_size(size)
                    })
            
            # 紀錄類別資訊
            total_files += cat_files
            total_size += cat_size
            
            report.append({
                'category': category,
                'files': cat_files,
                'size': cat_size,
                'size_str': cls._format_size(cat_size),
                'subcategories': sorted(subcat_info, key=lambda x: x['size'], reverse=True)
            })
        
        # 排序類別（按大小）
        report = sorted(report, key=lambda x: x['size'], reverse=True)
        
        # 輸出報告
        print("\n===== 快取使用報告 =====")
        print(f"總檔案數: {total_files}")
        print(f"總大小: {cls._format_size(total_size)}")
        print("-------------------------")
        
        for cat_info in report:
            if cat_info['files'] > 0:
                print(f"{cat_info['category']}: {cat_info['files']} 檔案, {cat_info['size_str']}")
                
                # 顯示子類別
                for subcat in cat_info['subcategories']:
                    print(f"  - {subcat['subcategory']}: {subcat['files']} 檔案, {subcat['size_str']}")
        
        print("=========================")
        
        # 建立DataFrame報告
        flat_report = []
        for cat_info in report:
            for subcat in cat_info['subcategories']:
                flat_report.append({
                    'category': cat_info['category'],
                    'subcategory': subcat['subcategory'],
                    'files': subcat['files'],
                    'size_bytes': subcat['size'],
                    'size': subcat['size_str']
                })
        
        # 如果沒有資料，加入一個空列
        if not flat_report:
            flat_report.append({
                'category': 'none',
                'subcategory': 'none',
                'files': 0,
                'size_bytes': 0,
                'size': '0 bytes'
            })
        
        report_df = pd.DataFrame(flat_report)
        
        # 儲存報告
        report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(cls.BASE_CACHE_DIR, f"cache_report_{report_time}.csv")
        report_df.to_csv(report_path, index=False)
        print(f"報告已儲存至: {report_path}")
        
        return report_df

# 初始化快取系統
def initialize_cache_system():
    """初始化快取系統，建立資料夾結構"""
    return CacheManager.initialize()

# 自動遷移舊的快取檔案
def migrate_old_cache(confirm=True):
    """遷移舊的快取檔案到新結構"""
    return CacheManager.migrate_old_cache(confirm=confirm)

# 產生快取使用報告
def generate_cache_report():
    """產生快取使用報告"""
    return CacheManager.generate_report()

# 清理快取
def clear_cache(category=None, subcategory=None, pattern="*", confirm=True):
    """清理指定類別的快取"""
    return CacheManager.clear(category, subcategory, pattern, confirm)

if __name__ == "__main__":
    # 初始化快取系統
    initialize_cache_system()
    
    # 遷移舊的快取檔案
    migrate_old_cache()
    
    # 產生報告
    generate_cache_report()