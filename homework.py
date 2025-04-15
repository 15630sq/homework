# -*- coding: utf-8 -*-
"""
智能数据工程课程存储系统
功能：
1. 存储任意格式文件（支持PDF/DOCX/TXT等）
2. 模糊内容搜索（支持任意位置匹配）
3. 分页浏览结果
4. 跨平台彩色高亮显示
5. 自动依赖安装
6. 信息抽取（从不同格式文件中提取文本内容）
"""

import os
import sqlite3
import sys
import re
from pathlib import Path

class DirectFileStorage:
    def __init__(self):
        self.storage_dir = "course_data"
        self._init_storage()

    def _init_storage(self):
        """初始化存储系统"""
        os.makedirs(f"{self.storage_dir}/files", exist_ok=True)
        self.conn = sqlite3.connect(f"{self.storage_dir}/course.db")

        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_name TEXT NOT NULL,
                stored_name TEXT NOT NULL,
                file_type TEXT NOT NULL,
                content TEXT,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def store_file(self, file_path: str) -> dict:
        """存储用户文件"""
        try:
            if not os.path.exists(file_path):
                return {"status": "error", "message": "文件不存在"}

            original_name = Path(file_path).name
            file_ext = Path(file_path).suffix[1:].lower()
            stored_name = f"{os.urandom(4).hex()}_{original_name}"
            dest_path = f"{self.storage_dir}/files/{stored_name}"

            with self.conn:
                # 复制文件到存储目录
                with open(file_path, "rb") as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())

                # 提取文本内容（确保返回非None）
                content = self._extract_file_content(file_path, file_ext) or ""

                # 存入SQLite数据库
                cursor = self.conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO files (original_name, stored_name, file_type, content)
                    VALUES (?, ?, ?, ?)
                    """,
                    (original_name, stored_name, file_ext, content)
                )
                file_id = cursor.lastrowid

            return {
                "status": "success",
                "file_id": file_id,
                "stored_path": dest_path
            }

        except sqlite3.Error as e:
            self.conn.rollback()
            return {
                "status": "error",
                "message": f"数据库操作失败: {str(e)}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"存储失败: {str(e)}"
            }

    def _extract_file_content(self, file_path: str, file_ext: str) -> str:
        """提取文件内容（严格返回空字符串，避免None）"""
        supported_ext = {'txt', 'md', 'csv', 'json', 'pdf', 'docx'}
        if file_ext not in supported_ext:
            print(f"警告：不支持的文件格式 {file_ext}，内容未提取")
            return ""

        try:
            if file_ext == 'pdf':
                return self._extract_pdf_text(file_path)
            elif file_ext == 'docx':
                return self._extract_docx_text(file_path)
            else:
                with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
                    return f.read()
        except Exception as e:
            print(f"内容提取失败 ({file_ext}): {str(e)}")
            return ""  # 异常时强制返回空字符串

    def _extract_pdf_text(self, file_path: str) -> str:
        """提取PDF文本"""
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""  # 确保PDF提取非None
            return text
        except Exception as e:
            print(f"PDF提取错误: {str(e)}")
            return ""

    def _extract_docx_text(self, file_path: str) -> str:
        """提取DOCX文本"""
        try:
            from docx import Document
            doc = Document(file_path)
            return "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
        except Exception as e:
            print(f"DOCX提取错误: {str(e)}")
            return ""

    def search_files(self, keyword: str, page: int = 1, per_page: int = 3) -> dict:
        """模糊搜索功能（按标点分割+智能高亮）"""
        try:
            cursor = self.conn.cursor()
            safe_keyword = re.escape(keyword)  # 转义特殊字符

            # 使用LIKE进行模糊匹配（提升性能）
            query = f"SELECT COUNT(*) FROM files WHERE content LIKE ? OR original_name LIKE ?"
            params = (f"%{safe_keyword}%", f"%{safe_keyword}%")
            cursor.execute(query, params)
            total = cursor.fetchone()[0]

            total_pages = max(1, (total + per_page - 1) // per_page)
            page = max(1, min(page, total_pages))
            offset = (page - 1) * per_page

            query = f"""
                SELECT id, original_name, file_type, content
                FROM files
                WHERE content LIKE ? OR original_name LIKE ?
                ORDER BY upload_time DESC
                LIMIT ? OFFSET ?
            """
            params = (f"%{safe_keyword}%", f"%{safe_keyword}%", per_page, offset)
            cursor.execute(query, params)

            results = []
            for row in cursor.fetchall():
                file_id, name, file_type, content = row
                content = content or ""  # 确保content非None

                # 构建模糊匹配模式
                pattern = re.escape(keyword)  # 避免特殊字符影响
                name_matches = re.search(pattern, name, re.IGNORECASE)
                previews = []

                # 处理文件内容中的匹配（按标点分割）
                if content:
                    pos = 0
                    text = content
                    delimiters = r'[。！？;；.,]'  # 支持中/英文标点
                    while pos < len(text):
                        # 查找关键词
                        match = re.search(pattern, text[pos:], re.IGNORECASE)
                        if not match:
                            break

                        match_start = pos + match.start()
                        match_end = pos + match.end()

                        # 查找下一个标点符号（从关键词结束后开始找）
                        next_punct = re.search(delimiters, text[match_end:])
                        if next_punct:
                            end_pos = match_end + next_punct.start() + 1  # 包含标点符号
                        else:
                            end_pos = len(text)  # 无标点则取到文本末尾

                        # 提取预览并高亮
                        preview = text[match_start:end_pos]
                        preview = re.sub(pattern, f"\033[31m{keyword}\033[0m", preview, flags=re.IGNORECASE)
                        previews.append(preview)

                        pos = end_pos  # 从标点符号后继续搜索，避免重复

                # 处理文件名匹配
                if name_matches:
                    highlighted_name = re.sub(pattern, f"\033[31m{keyword}\033[0m", name, flags=re.IGNORECASE)
                    previews = [f"文件名匹配: {highlighted_name}"] if not previews else previews

                results.append({
                    "id": file_id,
                    "name": name,
                    "type": file_type,
                    "match_count": len(previews),
                    "previews": previews
                })

            return {
                "results": results,
                "total": total,
                "total_pages": total_pages,
                "current_page": page
            }

        except sqlite3.Error as e:
            print(f"搜索错误: {str(e)}")
            return {
                "results": [],
                "total": 0,
                "total_pages": 0,
                "current_page": 0
            }

    def list_files(self) -> list:
        """列出所有文件"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, original_name, file_type, strftime('%Y-%m-%d %H:%M:%S', upload_time)
                FROM files
                ORDER BY upload_time DESC
            """)
            return cursor.fetchall()
        except sqlite3.Error as e:
            print(f"文件列表错误: {str(e)}")
            return []

    def delete_file(self, file_id: int) -> bool:
        """删除文件"""
        try:
            with self.conn:
                cursor = self.conn.cursor()
                cursor.execute("SELECT stored_name FROM files WHERE id = ?", (file_id,))
                stored_name = cursor.fetchone()
                if not stored_name:
                    return False

                file_path = f"{self.storage_dir}/files/{stored_name[0]}"
                if os.path.exists(file_path):
                    os.remove(file_path)

                cursor.execute("DELETE FROM files WHERE id = ?", (file_id,))
                return True
        except Exception as e:
            print(f"删除错误: {str(e)}")
            return False


def setup_environment():
    """初始化环境和依赖"""
    required = {
        'PyPDF2>=3.0.0',
        'python-docx>=0.8.11',
        'colorama>=0.4.6',
    }
    try:
        import pkg_resources
        installed = {f"{pkg.key}>={pkg.version}" for pkg in pkg_resources.working_set}
        missing = required - installed

        if missing:
            print("正在安装/升级依赖...")
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
    except Exception as e:
        print(f"依赖安装失败: {str(e)}")
        print("请手动运行: pip install PyPDF2 python-docx colorama")

    if sys.platform == "win32":
        import colorama
        colorama.init(autoreset=True)


def print_highlighted(text):
    """跨平台彩色打印"""
    if sys.platform == "win32":
        try:
            from colorama import AnsiToWin32
            print(AnsiToWin32(sys.stdout).write(text))
        except ImportError:
            print(re.sub(r'\033\[[0-9;]+m', '', text))
    else:
        print(text)


def main():
    setup_environment()
    storage = DirectFileStorage()

    while True:
        print("\n=== 智能文档存储与搜索系统 ===")
        print("1. 存储文件（支持PDF/DOCX/TXT等）")
        print("2. 搜索文件（关键词高亮）")
        print("3. 列出所有文件")
        print("4. 删除文件")
        print("0. 退出系统")

        choice = input("请选择操作 (0-4): ").strip()

        if choice == "1":
            file_path = input("请输入文件路径（支持拖拽文件）: ").strip('"')
            result = storage.store_file(file_path)
            if result["status"] == "success":
                print(f"✅ 存储成功！文件ID: {result['file_id']}")
            else:
                print(f"❌ 存储失败: {result['message']}")

        elif choice == "2":
            keyword = input("请输入搜索关键词: ").strip()
            if not keyword:
                print("关键词不能为空，请重新输入")
                continue

            result = storage.search_files(keyword)
            if result['total'] == 0:
                print("无匹配结果")
            else:
                for item in result['results']:
                    print(f"\n{'=' * 60}")
                    print(f"ID: {item['id']} | 文件名: {item['name']}")
                    print(f"类型: {item['type']} | 匹配数: {item['match_count']}")
                    print('-' * 60)
                    for i, preview in enumerate(item['previews'], 1):
                        print(f"\n匹配{i}:")
                        print_highlighted(preview)
                    print('=' * 60)

        elif choice == "3":
            files = storage.list_files()
            print("\n已存储文件列表：")
            print("ID | 文件名                | 类型 | 上传时间")
            for f in files:
                print(f"{f[0]:3} | {f[1]:20.20} | {f[2]:4} | {f[3]}")

        elif choice == "4":
            file_id = input("请输入要删除的文件ID: ").strip()
            if storage.delete_file(int(file_id)):
                print("✅ 文件删除成功")
            else:
                print("❌ 文件ID不存在或删除失败")

        elif choice == "0":
            print("感谢使用！系统已退出")
            break

        else:
            print("无效输入，请重新选择 (0-4)")


if __name__ == "__main__":
    main()