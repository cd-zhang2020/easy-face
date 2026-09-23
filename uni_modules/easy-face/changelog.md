## 1.0.1（2026-09-23）
### 新增功能
- 🚪 门禁多对多识别（1:N 闸机核验）：新增 `startFaceAccessActivity`，启动独立门禁 Activity，摄像头持续对全部已录入人脸做实时 1:N 检索，返回匹配用户与最佳相似度
- 🔎 实时 1:N 检索：`recognizeFaceRealtime` 支持对输入特征与本地库逐一比对余弦相似度，返回最佳匹配
- 📊 已录入人数统计：新增 `getFaceCount`，用于门禁人数提示与空库校验
- 📤 特征列表导出：新增 `getAllFaceFeatures`，返回当前所有已注册人脸特征（faceID + 128 维浮点数组 Base64），用于前端查看/同步/校验
- 🧩 底层支持：新增 `FaceAccessActivity` 门禁识别页面、`FaceFeatureEntry` 数据类与 `FaceRepository.getAllFeatureEntries()` 枚举接口，直接复用 MMKV 已存 Base64，避免重复编解码
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
