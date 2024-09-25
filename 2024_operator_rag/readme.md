基于运营商文本数据的知识库检索: https://www.datafountain.cn/competitions/1045/datasets

# 文档解析
## PyPDFLoader
逐页解析

### 少量PDF打印异常 print page_content 
UnicodeEncodeError: 'utf-8' codec can't encode characters in position 105-112: surrogates not allowed
PyPDFLoader 其中提取images 不知道如何实现
直接使用spliter, 然后\n 检索后返回时移除， 结果48
encode('utf-8', 'ignore').decode('utf-8')

### PyPDFLoader 提取images
https://www.bilibili.com/video/BV1da4y1r7Wz/?spm_id_from=333.337.search-card.all.click&vd_source=48d52af8eec19b4d0b941c4ceee34422
pip install rapidocr-onnxruntime
PyPDFLoader('path', extract_images=True)
images = loader.load()


## PDFPlumberLoader
存在分栏的结构不建议


## PyMuPDFLoader

## PDFMiner
将整个文档解析成一个完整的文本，

## marker
https://github.com/VikParuchuri/marker
# 表格处理
tabula 输出的表格看着还不错，但是直接splitter, 结果没法看，如何去检索存储表格文件是一个学问
思路
1. 直接基于dataframe 定制化输出，检索内容.

# 知识库生成

# 文本召回




# graphrag 实践
## ollama Embedding 服务 
安装ollama 
curl -fsSL https://ollama.com/install.sh | sh

ollama serve
ollama pull bge-large
从日志中找到服务接口地址

## 初始化项目
mkdir -p ./ragtest/input
文本语料放在input下，暂时只支持txt ,csv

### 初始化工程
python -m graphrag.index --init --root ./ragtest

## 源代码适配
按照这里修改Embedding相关的文件
https://gitcode.csdn.net/66c6d35513e4054e7e7ca0ab.html
local 检索报错：chunk_text 这里不需要修改


### 修改配置 settings.yaml
我使用的是ollama Embedding + Deepseek LLM
**LLM 部分**
```api_key: sk-0bexx
  type: openai_chat # or azure_openai_chat
  model: deepseek-chat
  model_supports_json: false # recommended if this is available for your model.
  max_tokens: 4096
  # request_timeout: 180.0
  api_base: https://api.deepseek.com/v1
```

**Embedding 部分**
```
    api_key: ollama
    type: openai_embedding # or azure_openai_embedding
    model: bge-large
    api_base: https://127.0.0.1:11434/api
```
这个url在ollama 启动shell的log查找，然后配上/api 即可，对了，如果是LLM部分，尾部添加的是/v1

# 构建索引
python -m graphrag.index --root ./ragtest
## 查询测试
python -m graphrag.query --root ./ragtest --method global "联通的股票代码是多少"
python -m graphrag.query --root ./ragtest --method global "董事会秘书是谁"
python -m graphrag.query --root ./ragtest --method local "2022年公司净资产"