# easy-face 人脸识别+活体检测插件

> 完全离线的人脸识别、人脸对比、静默活体检测 Uniapp UTS 插件。代码完全开源，可免费商用。所有 AI 推理在设备端完成，无需网络，不收集人脸敏感数据。

## 功能概览

- **人脸录入（拍照）**：通过摄像头采集人脸，自动提取 128 维特征向量并本地保存
- **人脸录入（图片）**：从 Base64 编码的图片中提取人脸特征
- **1:1 人脸验证**：拍照后与已录入人脸对比，支持自定义相似度阈值，可选静默活体检测
- **静默活体检测**：无需用户配合动作，单帧判断是否为真人（需 `mini_fasnet.tflite` 模型）
- **人脸特征管理**：查询、同步、删除本地人脸数据
- **完全离线**：所有功能无需网络，不收集不上传任何人脸数据

## 平台支持

| 平台 | 状态 |
|------|------|
| Android | ✅ 已实现（API 21+） |
| iOS | ⏳ 开发中 |
| HarmonyOS | ⏳ 开发中 |
| Web | ❌ 不支持 |
| 小程序 | ❌ 不支持 |

## 技术架构

```
ML Kit Face Detection (bundled) → ArcFaceMobileFaceNet (TFLite) 特征提取 → MiniFASNet (TFLite) 活体检测
                                                                          ↓
                                                                 MMKV 本地持久化 → 余弦相似度对比
```

- **人脸检测**：Google ML Kit Face Detection (bundled 模式，无需谷歌服务)
- **特征提取**：TensorFlow Lite + ArcFaceMobileFaceNet (128维特征向量)
- **活体检测**：TensorFlow Lite + MiniFASNetV1/V2 (来自 [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing))
- **数据存储**：腾讯 MMKV (高性能 KV 存储)

## 使用方法

### 1. 导入

```javascript
import {
    addFaceBySDKCamera,
    addFaceBySDKImage,
    faceVerify,
    livenessVerify,
    getFaceFeature,
    insertFaceFeature,
    deleteFaceFeature,
    startFaceVerifyActivity,
    startCameraXFaceVerifyActivity
} from '@/uni_modules/easy-face';
```

### 2. 人脸录入（拍照）

```javascript
addFaceBySDKCamera("user_12345", false, (result) => {
    if (result.code == 1) {
        console.log("录入成功！特征值: " + result.faceFeature);
    } else {
        console.log("录入失败: " + result.msg);
    }
});
```

### 3. 人脸录入（图片）

```javascript
addFaceBySDKImage("user_12345", base64ImageString, (result) => {
    console.log("code:", result.code, "msg:", result.msg);
});
```

### 4. 1:1 人脸验证（含活体检测）

```javascript
faceVerify("user_12345", 0.6, true, (result) => {
    console.log("验证结果:", result.code, "相似度:", result.similarity, "活体分:", result.liveness);
    // code=1: 验证通过
    // code=2: 相似度不足
    // code=11: 活体检测未通过
});
```

### 5. 纯活体检测

```javascript
livenessVerify((result) => {
    if (result.code == 1) {
        console.log("活体检测通过，活体分: " + result.liveness);
    }
});
```

### 6. 全屏人脸验证（Camera2 原生 UI）

启动原生全屏人脸验证 Activity，使用 Camera2 + 自定义 Canvas UI：

```javascript
startFaceVerifyActivity("user_12345", 0.6, true, (result) => {
    console.log("验证结果:", result.code, "相似度:", result.similarity, "活体分:", result.liveness);
});
```

### 7. 全屏人脸验证（CameraX 原生 UI）

启动基于 CameraX 的全屏人脸验证 Activity，自带精美的扫描线/脉冲光环/粒子爆炸动画效果：

```javascript
startCameraXFaceVerifyActivity("user_12345", 0.6, true, (result) => {
    console.log("验证结果:", result.code, "相似度:", result.similarity, "活体分:", result.liveness);
});
```

## API 参数说明

### addFaceBySDKCamera(faceID, needShowConfirmDialog, callback)

| 参数 | 类型 | 说明 |
|------|------|------|
| faceID | string | 用户唯一标识 |
| needShowConfirmDialog | boolean | 是否显示确认框（预留） |
| callback | function | `(result: ResultJSON) => void` |

### faceVerify(faceID, threshold, needLiveness, callback)

| 参数 | 类型 | 说明 |
|------|------|------|
| faceID | string | 用户唯一标识 |
| threshold | number | 相似度阈值 [0.5, 0.95]，默认 0.6 |
| needLiveness | boolean | 是否启用静默活体检测 |
| callback | function | `(result: ResultJSON) => void` |

### startFaceVerifyActivity(faceID, threshold, needLiveness, callback)

启动原生全屏人脸验证 Activity，使用 **Camera2 + 自定义 Canvas UI**，体验更好。

| 参数 | 类型 | 说明 |
|------|------|------|
| faceID | string | 用户唯一标识 |
| threshold | number | 相似度阈值 [0.5, 0.95] |
| needLiveness | boolean | 是否需要静默活体检测 |
| callback | function | `(result: ResultJSON) => void` |

### startCameraXFaceVerifyActivity(faceID, threshold, needLiveness, callback)

启动基于 **CameraX + 精美自定义 Canvas UI** 的全屏人脸验证 Activity。

相比 Camera2 版本的区别：
- 自动管理摄像头生命周期
- PreviewView 预览
- 自带扫描线 / 脉冲光环 / 粒子爆炸动画

| 参数 | 类型 | 说明 |
|------|------|------|
| faceID | string | 用户唯一标识 |
| threshold | number | 相似度阈值 [0.5, 0.95] |
| needLiveness | boolean | 是否需要静默活体检测 |
| callback | function | `(result: ResultJSON) => void` |

## ResultJSON 返回格式

```typescript
type ResultJSON = {
    code: number,         // 状态码
    msg: string,          // 消息
    faceID: string,       // 用户ID
    similarity: number,   // 相似度 0~1
    liveness: number,     // 活体分数 0~1
    faceFeature: string,  // 人脸特征值（Base64）
    faceBase64: string    // 人脸图 Base64
}
```

## 状态码说明

| 状态码 | 含义 |
|--------|------|
| 0 | 用户取消/操作中断 |
| 1 | 操作成功 |
| 2 | 人脸相似度过低（验证失败） |
| 5 | 多次未检测到人脸 |
| 6 | 本地无对应人脸特征值 |
| 11 | 静默活体检测失败 |
| 9010001 | 相机权限未授权 |
| 9010002 | 人脸检测失败 |
| 9010003 | 特征提取失败 |
| 9010004 | 活体检测失败 |
| 9010005 | AI模型加载失败 |
| 9010006 | 图片解码失败 |
| 9010007 | 数据存储异常 |
| 9010009 | 参数校验失败 |

## 模型文件说明

插件需要以下 TFLite 模型文件，放置在 `uni_modules/easy-face/utssdk/app-android/assets/` 目录下：

| 文件 | 用途 | 大小 | 来源 |
|------|------|------|------|
| `arcface_mobilefacenet.tflite` | 人脸特征提取 | ~4MB | [AcrfaceMobileFaceNet_TF] |
| `mini_fasnet.tflite` | 静默活体检测 | ~1.6MB | [Silent-Face-Anti-Spoofing-TFLite](https://github.com/feni-katharotiya/Silent-Face-Anti-Spoofing-TFLite) |

> **注意**：`mini_fasnet.tflite` 为可选项。未放置时活体检测功能将自动跳过，不影响人脸录入和比对功能。

## 开发调试

1. 下载本仓库到本地 `uni_modules/easy-face/`
2. 在 HBuilderX 中制作**自定义调试基座**
3. 使用自定义基座运行到手机
4. 调试阶段请确保 `AndroidManifest.xml` 中 `debuggable` 为 `true`

## 注意事项

- 人脸录入时请确保光线充足、人脸正面清晰
- 建议相似度阈值设置在 0.55 ~ 0.7 之间
- 活体检测模型需要单独下载，见上方模型文件说明
- 插件完全离线运行，所有人脸数据仅存储在设备本地

## 待办

- [ ] 人脸关键点的仿射变换对齐，提升大角度侧脸特征提取精度
- [ ] 动作活体检测（眨眼、张嘴、摇头等），增强活体安全等级

## License

Apache-2.0

## 特别声明

> ⚠️ **本项目代码全部由 AI 自动生成。**
>
> 本插件所使用的 AI 模型（人脸检测、特征提取、活体检测等）均来源于开源互联网项目，版权归原作者所有。
>
> 如有任何内容侵犯了您的合法权益，请联系作者进行删除处理。

## 致谢

| 项目 | 用途 | 开源协议 | 地址 |
|------|------|----------|------|
| Google ML Kit | 人脸检测 | Apache-2.0 | https://developers.google.com/ml-kit |
| ArcFaceMobileFaceNet | 人脸特征提取 | MIT | https://github.com/bubbliiiing/arcface-tf2 |
| Silent-Face-Anti-Spoofing | 静默活体检测 | Apache-2.0 | https://github.com/minivision-ai/Silent-Face-Anti-Spoofing |
| TensorFlow Lite | AI 推理引擎 | Apache-2.0 | https://github.com/tensorflow/tensorflow |
| AndroidX CameraX | 相机预览与分析 | Apache-2.0 | https://github.com/androidx/androidx |
| Tencent MMKV | 高性能 KV 存储 | Apache-2.0 | https://github.com/Tencent/MMKV |
