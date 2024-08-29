from fastapi import FastAPI, Request
# from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
# from peft import PeftModel
import uvicorn
import json
import datetime
import torch
from torchvision import models
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
# from peft import LoraConfig, PeftModel, TaskType
from pathlib import Path

# 设置设备参数
DEVICE = "cuda"  # 使用CUDA
DEVICE_ID = "0"  # CUDA设备ID，如果未设置则为空
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # 组合CUDA设备信息


# 清理GPU内存函数
def torch_gc():
    if torch.cuda.is_available():  # 检查是否可用CUDA
        with torch.cuda.device(CUDA_DEVICE):  # 指定CUDA设备
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片


# 构建 chat 模版
def build_input(image_path:str|Path)->torch.Tensor:
    input_image = Image.open(image_path)  # 替换为实际图片路径
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # 创建一个batch维度, [C, H, W] -> [B, C, H, W] , B=1

    # 将输入数据移动到GPU上
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
    return input_batch


# 创建FastAPI应用
app = FastAPI()


# 处理POST请求的端点
@app.post("/")
async def create_item(request: Request):
    global model, categories  # 声明全局变量以便在函数内部使用模型
    print(model)
    json_post_raw = await request.json()  # 获取POST请求的JSON数据
    json_post = json.dumps(json_post_raw)  # 将JSON数据转换为字符串
    json_post_list = json.loads(json_post)  # 将字符串转换为Python对象
    image_path = json_post_list.get("image_path")  # 获取请求中的提示
    print(image_path)
    # 调用模型进行对话生成
    input_batch = build_input(image_path=image_path)
    print(input_batch.shape)
    output = model(input_batch)
    print(output.shape)
    # 如果是一个分割任务，在得到output后不能像这样操作了；就是要用这个output绘制对应的mask图像，并且把mask图像保存在一个路径下，然后返回这个路径即可。
    probabilities = F.softmax(output, dim=1)
    # 获取最高概率对应的类别索引
    top_prob, top_catid = torch.topk(probabilities, 1)

    # 获取类别名称
    category = categories[top_catid.item()]
    print(f"预测类别: {category} (概率: {top_prob.item():.4f})")

    
    now = datetime.datetime.now()  # 获取当前时间
    time = now.strftime("%Y-%m-%d %H:%M:%S")  # 格式化时间为字符串
    # 构建响应JSON
    answer = {"response": output, "status": 200, "time": time}

    torch_gc()  # 执行GPU内存清理
    return answer  # 返回响应


# 主函数入口
if __name__ == "__main__":
    # 加载预训练模型对应的类别列表
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # 加载预训练的ResNet-18模型
    model = models.resnet18(weights='DEFAULT')  # 使用PyTorch提供的预训练权重
    model.eval()  # 设置为评估模式
    print("-" * 10, "加载 ResNet-18 模型成功", "-" * 10)
    # print(model)

    # 启动FastAPI应用
    # 用6006端口可以将autodl的端口映射到本地，从而在本地使用api
    uvicorn.run(app, host="0.0.0.0", port=6076, workers=1)  # 在指定端口和主机上启动应用
