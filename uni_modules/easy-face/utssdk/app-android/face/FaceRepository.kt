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
