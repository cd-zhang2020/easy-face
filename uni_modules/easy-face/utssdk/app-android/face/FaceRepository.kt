package uts.sdk.modules.easyFace.face

import android.content.Context
import android.graphics.Bitmap
import android.util.Base64
import android.util.Log
import com.tencent.mmkv.MMKV
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * 1:N 检索结果（供 UTS 层读取 faceID / score 字段）
 */
data class FaceMatch(val faceID: String, val score: Float)

/**
 * 单条已注册人脸（供 UTS 层 getAllFaceFeatures 接口遍历）
 * feature 为 MMKV 中已存储的 Base64 编码 128 维浮点数组，无需重复编解码
 */
data class FaceFeatureEntry(val faceID: String, val feature: String)

/**
 * 人脸数据仓库: 基于 MMKV 持久化存储人脸特征值
 * 支持余弦相似度对比
 */
class FaceRepository(context: Context) {

    companion object {
        private const val TAG = "EasyFace"
        private const val KEY_FACE_PREFIX = "easy_face_feature_"
    }

    private val mmkv: MMKV = MMKV.defaultMMKV()

    init {
        Log.d(TAG, "FaceRepository initialized with MMKV")
    }

    /**
     * 注册/保存人脸特征值
     * @param faceID 用户唯一标识
     * @param embedding 128维特征向量 (FloatArray)
     * @return 是否保存成功
     */
    fun saveFeature(faceID: String, embedding: FloatArray): Boolean {
        return try {
            val key = KEY_FACE_PREFIX + faceID
            val bytes = floatArrayToBytes(embedding)
            val base64Str = Base64.encodeToString(bytes, Base64.NO_WRAP)
            mmkv.encode(key, base64Str)
            Log.d(TAG, "Feature saved for faceID=$faceID, size=${embedding.size}")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save feature for faceID=$faceID", e)
            false
        }
    }

    /**
     * 获取人脸特征值
     * @param faceID 用户唯一标识
     * @return 128维特征向量，不存在返回 null
     */
    fun getFeature(faceID: String): FloatArray? {
        return try {
            val key = KEY_FACE_PREFIX + faceID
            val base64Str = mmkv.decodeString(key)
            if (base64Str.isNullOrEmpty()) {
                Log.d(TAG, "Feature not found for faceID=$faceID")
                return null
            }
            val bytes = Base64.decode(base64Str, Base64.NO_WRAP)
            bytesToFloatArray(bytes)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get feature for faceID=$faceID", e)
            null
        }
    }

    /**
     * 删除人脸特征值
     * @param faceID 用户唯一标识
     */
    fun deleteFeature(faceID: String) {
        val key = KEY_FACE_PREFIX + faceID
        mmkv.removeValueForKey(key)
        Log.d(TAG, "Feature deleted for faceID=$faceID")
    }

    /**
     * 判断人脸特征是否存在
     */
    fun isFeatureExist(faceID: String): Boolean {
        val key = KEY_FACE_PREFIX + faceID
        return !mmkv.decodeString(key).isNullOrEmpty()
    }

    /**
     * 获取已注册人脸总数（用于门禁 1:N 场景的人数提示与空库校验）
     */
    fun getRegisteredCount(): Int {
        return getAllFeatures().size
    }

    /**
     * 枚举全部已注册人脸特征（门禁 1:N 检索的数据源）
     * 仅取本插件前缀 easy_face_feature_ 的 key，剥离前缀得到 faceID
     * @return Map<faceID, 128维特征向量>
     */
    fun getAllFeatures(): Map<String, FloatArray> {
        val result = LinkedHashMap<String, FloatArray>()
        val allKeys = mmkv.allKeys() ?: return result
        for (key in allKeys) {
            if (key == null || !key.startsWith(KEY_FACE_PREFIX)) continue
            val faceID = key.substring(KEY_FACE_PREFIX.length)
            if (faceID.isEmpty()) continue
            val base64Str = mmkv.decodeString(key)
            if (base64Str.isNullOrEmpty()) continue
            try {
                val bytes = Base64.decode(base64Str, Base64.NO_WRAP)
                result[faceID] = bytesToFloatArray(bytes)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to decode feature for faceID=$faceID", e)
            }
        }
        Log.d(TAG, "getAllFeatures loaded ${result.size} faces")
        return result
    }

    /**
     * 枚举全部已注册人脸特征，直接返回 MMKV 中存储的 Base64 字符串（门禁 1:N 列表导出接口）
     * 相比 getAllFeatures 省去 Base64 解码再编码，导出大库时更高效
     * @return List<FaceFeatureEntry>，每项含 faceID 与特征 Base64 字符串
     */
    fun getAllFeatureEntries(): List<FaceFeatureEntry> {
        val result = ArrayList<FaceFeatureEntry>()
        val allKeys = mmkv.allKeys() ?: return result
        for (key in allKeys) {
            if (key == null || !key.startsWith(KEY_FACE_PREFIX)) continue
            val faceID = key.substring(KEY_FACE_PREFIX.length)
            if (faceID.isEmpty()) continue
            val base64Str = mmkv.decodeString(key)
            if (base64Str.isNullOrEmpty()) continue
            result.add(FaceFeatureEntry(faceID, base64Str))
        }
        Log.d(TAG, "getAllFeatureEntries loaded ${result.size} faces")
        return result
    }

    /**
     * 1:N 检索：将输入特征与全部已注册特征逐一比对余弦相似度，返回最佳匹配
     * 注：FaceNet 特征已 L2 归一化，余弦相似度等价于点积，开销为 O(N*128)
     * @param embedding 待检索的 128 维特征向量
     * @param threshold 相似度阈值 [0.5, 0.95]
     * @return FaceMatch；库为空时 faceID 为空串、score 为 -1，否则返回最佳匹配（可能低于阈值）
     */
    fun recognize(embedding: FloatArray, threshold: Float): FaceMatch {
        var bestID = ""
        var bestScore = -1f
        val all = getAllFeatures()
        for ((faceID, stored) in all) {
            val score = cosineSimilarity(embedding, stored)
            if (score > bestScore) {
                bestScore = score
                bestID = faceID
            }
        }
        // bestScore 为 -1 表示库为空；低于阈值视为陌生人（faceID 仍回传用于展示，由调用方按阈值判定）
        return if (bestID.isEmpty() || bestScore < 0f) {
            FaceMatch("", -1f)
        } else {
            FaceMatch(bestID, bestScore)
        }
    }

    /**
     * 计算两个特征向量的余弦相似度
     * @return 相似度 (0~1)
     */
    fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        if (a.size != b.size) return 0f
        var dot = 0f
        var na = 0f
        var nb = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        val denom = sqrt(na.toDouble()) * sqrt(nb.toDouble())
        return if (denom == 0.0) 0f else (dot / denom).toFloat()
    }

    /**
     * FloatArray → ByteArray
     */
    private fun floatArrayToBytes(floats: FloatArray): ByteArray {
        val buffer = ByteBuffer.allocate(floats.size * 4)
        buffer.order(ByteOrder.LITTLE_ENDIAN)
        for (f in floats) {
            buffer.putFloat(f)
        }
        return buffer.array()
    }

    /**
     * ByteArray → FloatArray
     */
    private fun bytesToFloatArray(bytes: ByteArray): FloatArray {
        val buffer = ByteBuffer.wrap(bytes)
        buffer.order(ByteOrder.LITTLE_ENDIAN)
        val floats = FloatArray(bytes.size / 4)
        for (i in floats.indices) {
            floats[i] = buffer.float
        }
        return floats
    }
}
