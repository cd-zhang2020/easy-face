package uts.sdk.modules.easyFace.face

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.graphics.*
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.sqrt

/**
 * FaceNet / MobileFaceNet 特征提取辅助类
 *
 * 需要模型: mobile_face_net.tflite
 *   - 输入: [1, 112, 112, 3] float32, 像素值归一化到 [-1, 1]
 *   - 输出: [1, embedding_dim] float32 特征向量
 *
 * 对齐流程:
 *   1. 根据边界框裁剪人脸区域
 *   2. 缩放到 112x112
 */
class FaceNetHelper(context: Context) {

    companion object {
        const val INPUT_SIZE = 112        // 模型输入尺寸
        const val EMBEDDING_SIZE = 128    // 特征向量维度
        private const val MODEL_PATH = "mobile_face_net.tflite"
        private const val TAG = "EasyFace"
    }

    /** 供 UTS 访问 EMBEDDING_SIZE */
    fun getEmbeddingSize(): Int = EMBEDDING_SIZE

    private val interpreter: Interpreter

    init {
        val options = Interpreter.Options().apply { setNumThreads(4) }
        val modelBuffer: ByteBuffer = loadModelFile(context, MODEL_PATH)
        interpreter = Interpreter(modelBuffer, options)
        Log.d(TAG, "FaceNet TFLite model loaded, embedding size=$EMBEDDING_SIZE")
    }

    private fun loadModelFile(context: Context, path: String): ByteBuffer {
        val afd: AssetFileDescriptor = context.assets.openFd(path)
        FileInputStream(afd.fileDescriptor).use { fis ->
            val channel = fis.channel
            return channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
        }
    }

    /**
     * 对齐并提取人脸特征向量
     * @param bitmap 原始图片
     * @param bbox 人脸边界框 (图像坐标系)
     * @param leftEye 左眼位置 (图像坐标系)
     * @param rightEye 右眼位置 (图像坐标系)
     * @return 归一化特征向量 FloatArray
     */
    fun extractEmbedding(
        bitmap: Bitmap,
        bbox: RectF,
        leftEye: PointF,
        rightEye: PointF
    ): FloatArray? {
        val aligned = alignFace(bitmap, bbox, leftEye, rightEye) ?: return null
        return runInference(aligned)
    }

    /**
     * 人脸对齐: 根据边界框裁剪 → 缩放到 112x112
     */
    private fun alignFace(
        bitmap: Bitmap,
        bbox: RectF,
        leftEye: PointF,
        rightEye: PointF
    ): Bitmap? {
        // 扩展边界框 (各方向加 15% margin)
        val bw = bbox.width()
        val bh = bbox.height()
        val marginX = bw * 0.15f
        val marginY = bh * 0.15f

        val cropLeft = (bbox.left - marginX).coerceAtLeast(0f)
        val cropTop = (bbox.top - marginY * 1.2f).coerceAtLeast(0f)
        val cropRight = (bbox.right + marginX).coerceAtMost(bitmap.width.toFloat())
        val cropBottom = (bbox.bottom + marginY * 0.5f).coerceAtMost(bitmap.height.toFloat())

        val cropW = (cropRight - cropLeft).toInt()
        val cropH = (cropBottom - cropTop).toInt()
        if (cropW <= 10 || cropH <= 10) return null

        val faceRegion = Bitmap.createBitmap(bitmap, cropLeft.toInt(), cropTop.toInt(), cropW, cropH)
        return Bitmap.createScaledBitmap(faceRegion, INPUT_SIZE, INPUT_SIZE, true)
    }

    /**
     * 将 Bitmap 转为 TFLite 输入 ByteBuffer 并推理
     */
    private fun runInference(bitmap: Bitmap): FloatArray? {
        val inputBuffer = bitmapToBuffer(bitmap)

        // 输出 buffer
        val outputSize = EMBEDDING_SIZE
        val outputBuffer = Array(1) { FloatArray(outputSize) }

        interpreter.run(inputBuffer, outputBuffer)

        val embedding = outputBuffer[0]
        // L2 归一化
        val norm = sqrt(embedding.map { it * it }.sum().toDouble()).toFloat()
        if (norm == 0f) return null
        for (i in embedding.indices) {
            embedding[i] /= norm
        }
        return embedding
    }

    /**
     * Bitmap → ByteBuffer (float32, [1, 112, 112, 3])
     * 归一化到 [-1, 1]: (pixel / 127.5) - 1.0
     * 通道顺序: RGB
     */
    private fun bitmapToBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
        buffer.order(ByteOrder.nativeOrder())
        buffer.rewind()

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (pixel in pixels) {
            buffer.putFloat(((pixel shr 16) and 0xFF) / 127.5f - 1.0f) // R
            buffer.putFloat(((pixel shr 8) and 0xFF) / 127.5f - 1.0f)   // G
            buffer.putFloat((pixel and 0xFF) / 127.5f - 1.0f)           // B
        }
        buffer.rewind()
        return buffer
    }

    fun close() {
        interpreter.close()
        Log.d(TAG, "FaceNet TFLite interpreter closed")
    }
}
