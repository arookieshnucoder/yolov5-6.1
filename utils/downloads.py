# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Download utils
    工具类，负责从github/googleleaps/google drive 等网站或者云服务器上下载所需的一些文件。
    这个文件比较重要的是两个函数：safe_download和attempt_download。在train.py或者yolo.py等文件中都会用到。
"""

"""
    ·注释来源于各位大佬的视频+博客，收集不易，祝你早日出sci！
    ·秉持开源精神！取之于大佬，用之于各位！
    ·@Dragon AI 
"""

import os
import platform
import subprocess
import time
import urllib
from pathlib import Path
from zipfile import ZipFile

import requests
import torch


def gsutil_getsize(url=''):
    """
        函数功能：返回网站链接url对应文件的大小。
        被调用：在google_utils.py的print_mutation函数
    """
    # gs://bucket/file size https://cloud.google.com/storage/docs/gsutil/commands/du
    # 创建一个子进程在命令行执行 gsutil du url 命令(访问 Cloud Storage) 返回执行结果(文件)
    s = subprocess.check_output(f'gsutil du {url}', shell=True).decode('utf-8')
    # 返回文件的bytes大小
    return eval(s.split(' ')[0]) if len(s) else 0  # bytes

# ===================================这两个函数主要是用来从github或者googleleaps云服务器中下载文件的，主要是下载权重文件===================================

def safe_download(file, url, url2=None, min_bytes=1E0, error_msg=''):
    """
        函数功能：下载 url（github）或者url2（googleleaps云服务器)路径对应的网页文件，通常是下载权重文件
        被调用：在attempt_download函数中
    """

    """
        :params file: 要下载的文件名
        :params url: 第一个下载地址 一般是github
        :params url2: 第二个下载地址(第一个下载地址下载失败后使用) 一般是googleleaps等云服务器
        :params min_bytes: 判断文件是否下载下来 只有文件存在且文件大小要大于min_bytes才能判断文件已经下载下来了
        :params error_msg: 文件下载失败的显示信息 初始化默认’‘
    """

    # Attempts to download file from url or url2, checks and removes incomplete downloads < min_bytes
    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1 # 尝试从url中下载文件 一般是github
        print(f'Downloading {url} to {file}...')
        # 从url中下载文件
        torch.hub.download_url_to_file(url, str(file))
        # 判断文件是否下载下来了(文件存在且文件大小要大于min_bytes)
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2 #尝试从url2中下载文件  一般是googleleaps(云服务器)
        # 移除之前下载失败的文件
        file.unlink(missing_ok=True)  # remove partial downloads
        print(f'ERROR: {e}\nRe-attempting {url2 or url} to {file}...')
        os.system(f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -")  # curl download, retry and resume on fail
    finally:
        # 检查文件是否下载下来了(是否存在) 或 文件大小是否小于min_bytes
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            # 下载失败 移除下载失败的文件 remove partial downloads
            file.unlink(missing_ok=True)  # remove partial downloads
            # 打印错误信息
            print(f"ERROR: {assert_msg}\n{error_msg}")
        print('')


def attempt_download(file, repo='ultralytics/yolov5'):  # from utils.downloads import *; attempt_download()
    """
        函数功能：实现从几个云平台(github/googleleaps云服务器)下载文件(预训练模型/权重)
        被调用：在experimental.py中的attempt_load函数和train.py
    """
    """
        :params file: 如果是文件路径 且这个路径不存在文件就尝试下载文件
                        果是url地址 就直接下载文件
                        如果只是一个要下载的文件名, 那就获取版本号开始下载(github/googleleaps)
        :params repo: 下载文件的github仓库名 默认是'ultralytics/yolov5'
    """

    # .strip()删除字符串前后空格 /n /t等  
    # .replace将 ' 替换为空格  Path将str转换为Path对象
    file = Path(str(file).strip().replace("'", ''))

     # 如果这个文件路径不存在文件 就尝试下载
    if not file.exists():
        # urllib.parse: 解析url   .unquote: 对url进行解码  decode '%2F' to '/' etc.
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
         # 如果解析的文件名是http:/ 或 https:/ 开头就直接下载
        if str(file).startswith(('http:/', 'https:/')):  # download
            # url: 下载路径 url
            url = str(file).replace(':/', '://')  # Pathlib turns :// -> :/
            # name: 要下载的文件名
            file = name.split('?')[0]  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                print(f'Found {url} locally at {file}')  # file already exists
            else:
                 # 下载文件
                safe_download(file=file, url=url, min_bytes=1E5)
            return file

        # GitHub assets
        file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
        try:
             # 利用github api 获取最新的版本相关信息  这里的response是一个打字典
            response = requests.get(f'https://api.github.com/repos/{repo}/releases/latest').json()  # github api
            # response['assets']中包含多个字典的列表 其中记录每一个asset的相关信息
            # release assets, i.e. ['yolov5s.pt', 'yolov5m.pt', ...]
            assets = [x['name'] for x in response['assets']]  # release assets, i.e. ['yolov5s.pt', 'yolov5m.pt', ...]
            # tag: 当前最新版本号 如'v5.0'           
            tag = response['tag_name']  # i.e. 'v1.0'
        except Exception:   # 获取失败 就退而求其次 直接利用git命令强行补齐版本信息
            assets = ['yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt',
                      'yolov5n6.pt', 'yolov5s6.pt', 'yolov5m6.pt', 'yolov5l6.pt', 'yolov5x6.pt']
            try:
                # 创建一个子进程在命令行执行 git tag 命令(返回版本号 版本号信息一般在字典最后 -1) 返回执行结果(版本号tag)
                tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
            except Exception:
                # 如果还是失败 就强行自己补一个版本号 tag='v6.0'
                tag = 'v6.0'  # current release

        if name in assets:
            # 开始从github中下载文件
            # file: 要下载的文件名
            # url: 第一个下载地址 一般是github  repo: github仓库名  tag: 版本号  name: 文件名 .pt
            # url2: 第二个备用的下载地址 一般是googleapis(云服务器)
            # min_bytes: 判断文件是否下载下来 只有文件存在且文件大小要大于min_bytes才能判断文件已经下载下来了
            # error_msg: 下载失败的显示信息
            safe_download(file,
                          url=f'https://github.com/{repo}/releases/download/{tag}/{name}',
                          # url2=f'https://storage.googleapis.com/{repo}/ckpt/{name}',  # backup url (optional)
                          min_bytes=1E5,
                          error_msg=f'{file} missing, try downloading from https://github.com/{repo}/releases/')

    return str(file)

# ============================================================================================================================================

# ===================================这两个函数实现从google drive上下载压缩文件并将其解压, 再删除掉压缩文件========================================

def gdrive_download(id='16TiPfZj7htmTyhntwcZyEEAejOUxuT6m', file='tmp.zip'):
    """
        函数功能：实现从google drive下载压缩文件并将其解压, 再删除掉压缩文件。
    """
    """
        :params id: url ?后面的id参数的参数值
        :params file: 需要下载的压缩文件名
    """
    # Downloads a file from Google Drive. from yolov5.utils.downloads import *; gdrive_download()
    t = time.time() # 获取当前时间
    file = Path(file)   # Path将str转换为Path对象
    cookie = Path('cookie')  # gdrive cookie
    print(f'Downloading https://drive.google.com/uc?export=download&id={id} as {file}... ', end='')
    file.unlink(missing_ok=True)  # 移除已经存在的文件(可能是下载失败/下载不完全)
    cookie.unlink(missing_ok=True)  # 移除已经存在的文件(可能是下载失败/下载不完全)

     # 尝试下载压缩文件
    out = "NUL" if platform.system() == "Windows" else "/dev/null"
    # 使用cmd命令从google drive上下载文件
    os.system(f'curl -c ./cookie -s -L "drive.google.com/uc?export=download&id={id}" > {out}')
    if os.path.exists('cookie'):  # large file
        # 如果文件较大 就需要有令牌get_token(存在cookie才有令牌)的指令s才能下载
        # get_token()函数在下面定义了 用于获取当前cookie的令牌token
        s = f'curl -Lb ./cookie "drive.google.com/uc?export=download&confirm={get_token()}&id={id}" -o {file}'
    else:  # small file
         # 小文件就不需要带令牌的指令s 直接下载就行
        s = f'curl -s -L -o {file} "drive.google.com/uc?export=download&id={id}"'
    # 执行下载指令s 并获得返回 如果cmd命令执行成功 则os.system()命令会返回0
    r = os.system(s)  # execute, capture return
    cookie.unlink(missing_ok=True)  # 再次移除已经存在的cookie

    # 下载错误检测  如果r != 0 则下载错误
    if r != 0:
        file.unlink(missing_ok=True)  # 下载错误 移除下载的文件(可能不完全或者下载失败)
        print('Download error ')  # raise Exception('Download error')
        return r

    # 如果是压缩文件 就解压  file.suffix方法可以获取file文件的后缀
    if file.suffix == '.zip':
        print('unzipping... ', end='')
        ZipFile(file).extractall(path=file.parent)  # unzip
        file.unlink()   # 移除.zip压缩文件

    print(f'Done ({time.time() - t:.1f}s)')  # 打印下载 + 解压过程所需要的时间
    return r    


def get_token(cookie="./cookie"):
    """
       函数功能：从cookie中获取令牌token。会在gdrive_download中被调用。
    """
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return ""

# ============================================================================================================================================


# Google utils: https://cloud.google.com/storage/docs/reference/libraries ----------------------------------------------
#
#
# def upload_blob(bucket_name, source_file_name, destination_blob_name):
#     # Uploads a file to a bucket
#     # https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python
#
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#
#     blob.upload_from_filename(source_file_name)
#
#     print('File {} uploaded to {}.'.format(
#         source_file_name,
#         destination_blob_name))
#
#
# def download_blob(bucket_name, source_blob_name, destination_file_name):
#     # Uploads a blob from a bucket
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#
#     blob.download_to_filename(destination_file_name)
#
#     print('Blob {} downloaded to {}.'.format(
#         source_blob_name,
#         destination_file_name))
