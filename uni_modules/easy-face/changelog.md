## 1.0.0 (2026-05-17)

### 新增功能
- 🎉 首次发布，完成 Android 平台基础功能
- 📷 人脸录入：支持通过摄像头拍照录入人脸
- 🖼️ 图片录入：支持从 Base64 图片提取人脸特征
- 🔍 1:1 人脸验证：拍照后与已录入人脸进行余弦相似度对比
- 🛡️ 静默活体检测：集成 MiniFASNet 模型，无需用户配合动作
- 💾 人脸特征管理：基于 MMKV 的本地存储，支持查询、同步、删除
- 🏗️ 技术栈：ML Kit Face Detection + TFLite ArcFaceMobileFaceNet + MiniFASNet

### 已知限制
- iOS 和 HarmonyOS 平台待实现
