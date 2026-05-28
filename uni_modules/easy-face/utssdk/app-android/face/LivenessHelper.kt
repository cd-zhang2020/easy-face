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

/**
 * MiniFASNet 静默活体检测辅助类
 *
 * 基于 Silent-Face-Anti-Spoofing (小视科技)
 * 模型: mini_fasnet.tflite
 *   - 输入: [1, 80, 80, 3] float32, BGR, 像素值 [0, 255]
 *   - 输出: [1, 3] float32 logits [spoof_print, real, spoof_replay]
 *
 * 模型下载: https://github.com/feni-katharotiya/Silent-Face-Anti-Spoofing-TFLite
 * 放到 utssdk/app-android/assets/ 目录下
 */
class LivenessHelper(context: Context) {

    companion object {
        private const val TAG = "EasyFace"
        const val INPUT_SIZE = 80          // MiniFASNet 输入尺寸
        private const val MODEL_PATH = "mini_fasnet.tflite"
        private const val LIVENESS_THRESHOLD = 0.5f  // 活体判定阈值
    }

    private val interpreter: Interpreter
    private val isModelLoaded: Boolean

    init {
        val tempInterpreter = try {
            val options = Interpreter.Options().apply { setNumThreads(2) }
               
            val modelBuffer = loadModelFile(context, MODEL_PATH)
            Interpreter(modelBuffer, options)
        } catch (e: Exception) {
            Log.w(TAG, "MiniFASNet model not found: $MODEL_PATH. Liveness detection disabled.", e)
            null
        }
        interpreter = tempInterpreter ?: Interpreter(ByteBuffer.allocateDirect(0))
        isModelLoaded = tempInterpreter != null

        if (isModelLoaded) {
            Log.d(TAG, "MiniFASNet TFLite model loaded, input size=$INPUT_SIZE")
        }
    }

    /**
     * 检查模型是否成功加载
     */
    fun isAvailable(): Boolean = isModelLoaded

    private fun loadModelFile(context: Context, path: String): ByteBuffer {
        val afd: AssetFileDescriptor = context.assets.openFd(path)
        FileInputStream(afd.fileDescriptor).use { fis ->
            val channel = fis.channel
            return channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
        }
    }

    /**
     * 对单张人脸图进行活体检测
     * @param bitmap 原始相机画面
     * @param bbox 人脸边界框
     * @return 活体置信度 (0~1)，越高越像真人；模型未加载时返回 -1f
     */
    fun detectLiveness(bitmap: Bitmap, bbox: RectF): Float {
        Log.d(TAG, "detectLiveness entry: bitmap=${bitmap.width}x${bitmap.height} recycled=${bitmap.isRecycled} bbox=${bbox}")
        if (!isModelLoaded) {
            Log.w(TAG, "Liveness model not loaded, skipping detection")
            return -1f
        }

        val faceCrop = cropFace(bitmap, bbox) ?: run {
            Log.w(TAG, "Failed to crop face for liveness, bitmap recycled=${bitmap.isRecycled}", Exception("cropFace trace"))
            return -1f
        }
        return runInference(faceCrop)
    }

    /**
     * 裁剪人脸区域并缩放到模型输入尺寸
     */
    private fun cropFace(bitmap: Bitmap, bbox: RectF): Bitmap? {
        val bw = bbox.width()
        val bh = bbox.height()
        Log.d(TAG, "cropFace bbox: left=${bbox.left} top=${bbox.top} right=${bbox.right} bottom=${bbox.bottom} w=$bw h=$bh")
        Log.d(TAG, "cropFace bitmap: w=${bitmap.width} h=${bitmap.height}")

        val marginX = bw * 0.1f
        val marginY = bh * 0.1f

        val cropLeft = (bbox.left - marginX).coerceAtLeast(0f)
        val cropTop = (bbox.top - marginY).coerceAtLeast(0f)
        val cropRight = (bbox.right + marginX).coerceAtMost(bitmap.width.toFloat())
        val cropBottom = (bbox.bottom + marginY).coerceAtMost(bitmap.height.toFloat())

        val cropW = (cropRight - cropLeft).toInt()
        val cropH = (cropBottom - cropTop).toInt()
        Log.d(TAG, "cropFace region: left=$cropLeft top=$cropTop right=$cropRight bottom=$cropBottom w=$cropW h=$cropH")
        if (cropW <= 10 || cropH <= 10) return null

        val faceRegion = Bitmap.createBitmap(bitmap, cropLeft.toInt(), cropTop.toInt(), cropW, cropH)
        return Bitmap.createScaledBitmap(faceRegion, INPUT_SIZE, INPUT_SIZE, true)
    }

    /**
     * MiniFASNet TFLite 推理
     * 输入: [1, 80, 80, 3] float32, RGB, 归一化到 [-1, 1]
     * 输出: [1, 3] float32 logits [real, spoof_print, spoof_replay]
     */
    private fun runInference(bitmap: Bitmap): Float {
        val inputBuffer = bitmapToBuffer(bitmap)
        val outputBuffer = Array(1) { FloatArray(3) }

        try {
            interpreter.run(inputBuffer, outputBuffer)
        } catch (e: Exception) {
            Log.e(TAG, "TFLite inference failed", e)
            return -1f
        }

        val scores = outputBuffer[0]
        // softmax: 将 logits 转为概率，三者之和 = 1
        val maxScore = maxOf(scores[0], scores[1], scores[2])
        val exp0 = Math.exp((scores[0] - maxScore).toDouble())
        val exp1 = Math.exp((scores[1] - maxScore).toDouble())
        val exp2 = Math.exp((scores[2] - maxScore).toDouble())
        val sumExp = exp0 + exp1 + exp2
        // index 0 = spoof_print, index 1 = real, index 2 = spoof_replay
        val spoofPrintProb = (exp0 / sumExp).toFloat()
        val realProb = (exp1 / sumExp).toFloat()
        val spoofReplayProb = (exp2 / sumExp).toFloat()
        Log.d(TAG, "liveness logits=[${scores[0]}, ${scores[1]}, ${scores[2]}] → softmax=[real=$realProb, print=$spoofPrintProb, replay=$spoofReplayProb] sum=${realProb + spoofPrintProb + spoofReplayProb}")
        return realProb
    }

    /**
     * Bitmap → ByteBuffer (float32, [1, 80, 80, 3])
     * MiniFASNet TFLite 模型预处理:
     *   通道: BGR (该 TFLite 转换使用 cv2.imread, OpenCV 默认 BGR)
     *   值域: [0, 255] float32 (inference_tflite.py 中仅 astype(np.float32)，未做 /255)
     */
    private fun bitmapToBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
        buffer.order(ByteOrder.nativeOrder())
        buffer.rewind()

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF).toFloat()
            val g = ((pixel shr 8) and 0xFF).toFloat()
            val b = (pixel and 0xFF).toFloat()
            // BGR 顺序 + [0, 255] 值域（与 inference_tflite.py 一致）
            buffer.putFloat(b) // B
            buffer.putFloat(g) // G
            buffer.putFloat(r) // R
        }

        buffer.rewind()
        return buffer
    }

    fun close() {
        if (isModelLoaded) {
            interpreter.close()
            Log.d(TAG, "MiniFASNet TFLite interpreter closed")
        }
    }
}
