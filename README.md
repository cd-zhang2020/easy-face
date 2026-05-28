# face-demo — uni-app x 人脸识别 Demo

基于 **uni-app x** 的离线人脸识别完整示例项目，展示如何使用 **easy-face UTS 插件** 实现人脸录入、1:1 验证、静默活体检测等功能。

> 本仓库同时包含 **easy-face UTS 插件源码**（位于 `uni_modules/easy-face/`），可供学习和二次开发。插件代码完全开源，Apache-2.0 协议，可免费商用。

## 仓库说明

| 路径 | 说明 |
|------|------|
| `uni_modules/easy-face/` | **easy-face UTS 插件源码** — 离线人脸识别 + 活体检测 |
| `pages/index/index.uvue` | 示例 Demo 页面 — 所有 API 的 UI 调用演示 |
| 其他文件 | uni-app x 标准工程文件 |

> 关于 easy-face 插件的详细介绍、API 文档、状态码说明、模型下载等，请查阅 [uni_modules/easy-face/readme.md](./uni_modules/easy-face/readme.md)。

## easy-face 插件简介

**easy-face** 是一个完全离线的人脸识别 + 静默活体检测 UTS 插件。所有 AI 推理在设备端完成，无需网络，不收集人脸敏感数据。

### 核心功能

- **人脸录入（拍照）**：摄像头采集人脸，自动提取 128 维特征向量并本地保存
- **人脸录入（图片）**：从 Base64 编码的图片中提取人脸特征
- **1:1 人脸验证**：拍照后与已录入人脸对比，可选静默活体检测
- **静默活体检测**：无需用户配合动作，单帧判断是否为真人
- **全屏原生验证**：Camera2 / CameraX 双模式全屏人脸验证 Activity，带精美动画
- **人脸特征管理**：查询、同步、删除本地人脸数据
- **完全离线**：不收集、不上传任何人脸数据

### 平台支持

| 平台 | 状态 |
|------|------|
| Android | ✅ 已实现（API 21+） |
| iOS | ⏳ 开发中 |
| HarmonyOS | ⏳ 开发中 |
| Web | ❌ 不支持 |
| 小程序 | ❌ 不支持 |

### 技术架构

```
ML Kit Face Detection (bundled) → ArcFaceMobileFaceNet (TFLite) 特征提取 → MiniFASNet (TFLite) 活体检测
                                                                         ↓
                                                                MMKV 本地持久化 → 余弦相似度对比
```

- **人脸检测**：Google ML Kit Face Detection（bundled 模式，无需谷歌服务）
- **特征提取**：TensorFlow Lite + ArcFaceMobileFaceNet（128 维特征向量）
- **活体检测**：TensorFlow Lite + MiniFASNetV1/V2
- **数据存储**：腾讯 MMKV

## Demo 演示

示例 Demo 提供了可视化的 UI 界面，可以直接在手机端体验全部功能：

| 功能模块 | 说明 |
|---------|------|
| 基础配置 | 设置用户 ID、相似度阈值、是否启用活体检测 |
| 人脸录入 | 摄像头拍照录入 / 相册图片录入 |
| 人脸验证 | 1:1 验证 / Camera2 全屏验证 / CameraX 动画增强验证 / 纯活体检测 |
| 特征管理 | 查询 / 同步 / 删除人脸特征 |
| 结果展示 | 状态码、相似度百分比、活体分数、人脸截图预览 |

### Demo 截图

<img src="./static/logo.png" width="80" />

> 完整截图请运行项目后查看。

## 快速开始

### 环境要求

- [HBuilderX](https://www.dcloud.io/hbuilderx.html)（最新版）
- Android 设备（API 21+）或模拟器
- （可选）`mini_fasnet.tflite` 活体检测模型

### 模型文件准备

在运行前，需要将 TFLite 模型文件放入 `uni_modules/easy-face/utssdk/app-android/assets/`：

| 文件 | 用途 | 必需 |
|------|------|------|
| `arcface_mobilefacenet.tflite` | 人脸特征提取 | ✅ 必需 |
| `mini_fasnet.tflite` | 静默活体检测 | 可选 |

> 模型下载地址见 [easy-face 插件文档](./uni_modules/easy-face/readme.md#模型文件说明)。

### 运行步骤

```bash
# 1. 克隆仓库
git clone <your-repo-url> face-demo
cd face-demo

# 2. 下载 TFLite 模型文件到指定目录
# 参考 uni_modules/easy-face/readme.md 中的模型文件说明

# 3. 使用 HBuilderX 打开项目
# 4. 制作自定义调试基座（运行 → 制作自定义调试基座）
# 5. 使用自定义基座运行到 Android 设备
```

## 项目结构

```
face-demo/
├── App.uvue                          # 应用入口
├── main.uts                          # 应用初始化
├── manifest.json                     # uni-app x 配置
├── pages.json                        # 页面路由配置
├── pages/
│   └── index/
│       └── index.uvue                # Demo 主页面（所有 API 调用示例）
├── static/
│   └── logo.png                      # 应用 Logo
├── uni_modules/
│   └── easy-face/                    # ★ UTS 插件源码
│       ├── readme.md                 # 插件完整文档
│       └── utssdk/
│           ├── interface.uts         # 插件接口定义（所有 API 声明）
│           └── app-android/          # Android 原生实现（Kotlin）
│               ├── config.json
│               ├── AndroidManifest.xml
│               └── assets/           # 模型文件目录
└── unpackage/                        # 构建产物
```

## Demo 代码示例

### 人脸录入

```javascript
import { addFaceBySDKCamera } from '@/uni_modules/easy-face';

addFaceBySDKCamera("user_12345", (result) => {
    if (result.code == 1) {
        console.log("录入成功！");
    } else {
        console.log("录入失败: " + result.msg);
    }
});
```

### 1:1 人脸验证（含活体）

```javascript
import { faceVerify } from '@/uni_modules/easy-face';

faceVerify("user_12345", 0.6, true, (result) => {
    // code=1: 验证通过 | code=2: 相似度不足 | code=11: 活体检测未通过
    console.log("相似度:", result.similarity, "活体分:", result.liveness);
});
```

### 全屏原生验证（CameraX 动画增强版）

```javascript
import { startCameraXFaceVerifyActivity } from '@/uni_modules/easy-face';

startCameraXFaceVerifyActivity("user_12345", 0.6, true, (result) => {
    console.log("验证结果:", result.code);
});
```

## 注意事项

- 本项目为 **uni-app x**（uts 模式）工程，需使用 HBuilderX 打开
- 人脸录入时请确保光线充足、人脸正面清晰
- 建议相似度阈值设置在 0.55 ~ 0.7 之间
- 插件完全离线运行，所有人脸数据仅存储在设备本地

## License

Apache-2.0

## 特别声明

> ⚠️ **easy-face 插件代码全部由 AI 自动生成。**
>
> 插件所使用的 AI 模型（人脸检测、特征提取、活体检测等）均来源于开源互联网项目，版权归原作者所有。
> 如有任何内容侵犯了您的合法权益，请联系作者进行删除处理。

## 致谢

| 项目 | 用途 | 开源协议 |
|------|------|----------|
| Google ML Kit | 人脸检测 | Apache-2.0 |
| ArcFaceMobileFaceNet | 人脸特征提取 | MIT |
| Silent-Face-Anti-Spoofing | 静默活体检测 | Apache-2.0 |
| TensorFlow Lite | AI 推理引擎 | Apache-2.0 |
| AndroidX CameraX | 相机预览与分析 | Apache-2.0 |
| Tencent MMKV | 高性能 KV 存储 | Apache-2.0 |
