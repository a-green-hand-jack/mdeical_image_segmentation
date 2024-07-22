# mdeical_image_segmentation
为了2024iGem-USTC -software 而建立，为了实现医学影像的分割的任务


## 时间安排
|内容|时间|
|--|--|
|文献阅读（Jack）|7-15 上午|
|端django框架统一（scs）|7-15 上午|
|pytorch基础知识（wck）|7-15 下午|
|线性神经网络（wck）|7-16 下午|
|多层感知机（wck）|7-17 下午|
|huggingface文档教学（Jack）|7-18 上午|
|卷积神经网络（wck）|7-18 下午|
|注意力机制（暂定）|···|


## 进一步的安排

### 实现 model 的部署

参考[llm的api的部署的实现方法](https://github.com/datawhalechina/self-llm/blob/dev1.1/models/LLaMA3/01-LLaMA3-8B-Instruct%20FastApi%20%E9%83%A8%E7%BD%B2%E8%B0%83%E7%94%A8.md)，这样的好处在于，第一，可以和现有的方法更好的结合在一起，这是一个通用的方法；第二，我们已经使用了 *transformers* 这样也更加的方便。

### 得到更合适的 model 并进行改造

之前分享了一些 paper 和 dataset ，其中的 [SAM-2D](https://github.com/OpenGVLab/SAM-Med2D) 是一个相对比较新的而且合适的工作，需要用我们的方式实现它，把它接入到我们的体系中；在这个模型之外，我们也需要研究怎么使用它提供的数据集[SA-Med2D-20M ](https://huggingface.co/datasets/OpenGVLab/SA-Med2D-20M)，看看这个要怎么用。

总之，我们需要确定：

1. 现在更好、使用参数更少的 model
2. 这些 model 使用了什么样的数据集
3. 为了测试这些 model 使用那些指标(f1/iou/...)
4. 这些 model 使用什么样的 loss funcation

### 说明

我们首先关注的是**2D**的model，因为它们消耗的计算资源相对比较小；**3D**的model等我们在2D上积累的足够的经验后再测试。

同时，我们不生产（应该）model，我们只是model的搬运工，所以精力应该是**改造model成为我们需要的形状**而不是**建立一个新的model**！


### 时间线

1. 在8月前实现 **unet-2D** 的部署，也就是在服务器（autodl）上部署好api，这样在网页上就是通过调用api实现分割的功能，参考：[llm的api的部署的实现方法](https://github.com/datawhalechina/self-llm/blob/dev1.1/models/LLaMA3/01-LLaMA3-8B-Instruct%20FastApi%20%E9%83%A8%E7%BD%B2%E8%B0%83%E7%94%A8.md)。
2. 在8月7号前，改造、测试，也就是加载已经训练好的 model 的参数，3种筛选好的 model ，然后部署到 autodl 上，测试能不能实现这个功能：用户选择不同的 model 来执行不同的分割的任务。
3. 没了😘

## 相关资源
1. [动手学深度学习](https://zh.d2l.ai/chapter_preface/index.html)
2. [huggingface文档](https://huggingface.co/docs/transformers/v4.42.0/en/trainer)
3. [大模型修炼之道(一): Transformer Attention is all you need](https://www.bilibili.com/video/BV1FH4y157ZC?vd_source=96a3f9090e2330e11bb6eff837ccbd50)
4. [unet3+](https://arxiv.org/pdf/2004.08790)
5. [llm的api的部署的实现方法](https://github.com/datawhalechina/self-llm/blob/dev1.1/models/LLaMA3/01-LLaMA3-8B-Instruct%20FastApi%20%E9%83%A8%E7%BD%B2%E8%B0%83%E7%94%A8.md)


