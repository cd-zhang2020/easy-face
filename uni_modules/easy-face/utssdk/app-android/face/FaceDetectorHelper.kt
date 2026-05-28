package uts.sdk.modules.easyFace.face

import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF
import android.util.Base64
import android.util.Log
import android.media.Image
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * ML Kit 人脸检测辅助类
 * - bundled 模式，不需要谷歌框架
 * - 返回人脸框 + 五官关键点 + 面部轮廓点
 */
class FaceDetectorHelper {

    companion object {
        private const val TAG = "EasyFace"

        // 人脸关键点常量
        const val LEFT_EYE = 0
        const val RIGHT_EYE = 1
        const val NOSE_BASE = 2
        const val BOTTOM_MOUTH = 3
        const val LEFT_MOUTH = 6
        const val RIGHT_MOUTH = 7

        // 轮廓常量
        const val CONTOUR_FACE = 1
        const val CONTOUR_LEFT_EYE = 6
        const val CONTOUR_RIGHT_EYE = 7
        const val CONTOUR_NOSE_BRIDGE = 13

        /**
         * FloatArray → Base64 编码
         */
        fun floatArrayToBase64(floats: FloatArray): String {
            val buffer = ByteBuffer.allocate(floats.size * 4)
            buffer.order(ByteOrder.LITTLE_ENDIAN)
            for (f in floats) { buffer.putFloat(f) }
            return Base64.encodeToString(buffer.array(), Base64.NO_WRAP)
        }

        /**
         * Base64 → FloatArray
         */
        fun base64ToFloatArray(b64: String): FloatArray {
            val bytes = Base64.decode(b64, Base64.NO_WRAP)
            val buffer = ByteBuffer.wrap(bytes)
            buffer.order(ByteOrder.LITTLE_ENDIAN)
            val floats = FloatArray(bytes.size / 4)
            for (i in floats.indices) { floats[i] = buffer.float }
            return floats
        }
    }

    private var detector: FaceDetector

    init {
        // 关闭 Firebase ML Kit 的遥测上报（解决国内连接超时问题）
        System.setProperty("com.google.firebase.FirebaseApp.allowDefaultApp", "false")
    
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            //检测关键点
            //.setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            //检测轮廓
            //.setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
            //最小人脸占比 0.3
            .setMinFaceSize(0.3f)
            .build()
        detector = FaceDetection.getClient(options)
        Log.d(TAG, "ML Kit FaceDetector initialized")
    }

    /**
     * 对 Bitmap 进行人脸检测，返回 ML Kit 原生 Face 列表
     */
    fun detect(bitmap: Bitmap): List<Face> {
        
        val image = InputImage.fromBitmap(bitmap, 0)
        return Tasks.await(detector.process(image))
    }

    /**
     * 【UTS 专用】检测最佳人脸，返回可直接使用的坐标数据
     * 
     * 返回 Map:
     *   "left" / "top" / "right" / "bottom" : Float  — 人脸框
     *   "leftEyeX" / "leftEyeY" : Float              — 左眼坐标
     *   "rightEyeX" / "rightEyeY" : Float            — 右眼坐标
     * 
     * @return 未检测到人脸时返回 null
     */
    fun detectBestFace(bitmap: Bitmap): Map<String, Float>? {
        val faces = detect(bitmap)
        if (faces.isEmpty()) return null

        val face = faces[0]
        val bbox = face.boundingBox

        return mapOf(
            "left" to bbox.left.toFloat(), "top" to bbox.top.toFloat(),
            "right" to bbox.right.toFloat(), "bottom" to bbox.bottom.toFloat(),
            "leftEyeX" to 0f, "leftEyeY" to 0f,
            "rightEyeX" to 0f, "rightEyeY" to 0f)

//
//        // 找左眼和右眼关键点 (使用 companion 常量: 0=LEFT_EYE, 1=RIGHT_EYE)
//        var leftEyePos: PointF? = null
//        var rightEyePos: PointF? = null
//        for (lm in face.allLandmarks) {
//            when (lm.landmarkType) {
//                LEFT_EYE -> leftEyePos = PointF(lm.position.x, lm.position.y)
//                RIGHT_EYE -> rightEyePos = PointF(lm.position.x, lm.position.y)
//            }
//        }
//
//        if (leftEyePos == null || rightEyePos == null) return null
//
//        return mapOf(
//            "left" to bbox.left.toFloat(), "top" to bbox.top.toFloat(),
//            "right" to bbox.right.toFloat(), "bottom" to bbox.bottom.toFloat(),
//            "leftEyeX" to leftEyePos.x.toFloat(), "leftEyeY" to leftEyePos.y.toFloat(),
//            "rightEyeX" to rightEyePos.x.toFloat(), "rightEyeY" to rightEyePos.y.toFloat()
//        )
    }

    /**
     * 【高性能】直接从 Camera2 Image (YUV_420_888) 异步检测人脸
     * 跳过 YUV→Bitmap 转换，零拷贝传给 ML Kit，节省 50-80ms
     *
     * @param image Camera2 原始 Image（YUV_420_888 格式），由调用方管理生命周期
     * @param rotationDegrees 图像旋转角度 (0/90/180/270)
     * @param callback 检测完成回调，在后台线程执行
     */
    fun detectBestFaceFromImage(
        image: Image,
        rotationDegrees: Int,
        callback: (Map<String, Float>?) -> Unit
    ) {
        val inputImage = InputImage.fromMediaImage(image, rotationDegrees)
        detector.process(inputImage)
            .addOnSuccessListener { faces ->
                try {
                    if (faces.isEmpty()) {
                        callback(null)
                        return@addOnSuccessListener
                    }
                    val face = faces[0]
                    val bbox = face.boundingBox

                    var leftEyePos: PointF? = null
                    var rightEyePos: PointF? = null
                    for (lm in face.allLandmarks) {
                        when (lm.landmarkType) {
                            LEFT_EYE -> leftEyePos = PointF(lm.position.x, lm.position.y)
                            RIGHT_EYE -> rightEyePos = PointF(lm.position.x, lm.position.y)
                        }
                    }

                    if (leftEyePos == null || rightEyePos == null) {
                        callback(null)
                        return@addOnSuccessListener
                    }

                    callback(mapOf(
                        "left" to bbox.left.toFloat(), "top" to bbox.top.toFloat(),
                        "right" to bbox.right.toFloat(), "bottom" to bbox.bottom.toFloat(),
                        "leftEyeX" to leftEyePos.x.toFloat(), "leftEyeY" to leftEyePos.y.toFloat(),
                        "rightEyeX" to rightEyePos.x.toFloat(), "rightEyeY" to rightEyePos.y.toFloat()
                    ))
                } catch (e: Exception) {
                    Log.e(TAG, "detectBestFaceFromImage: unexpected error in callback", e)
                    callback(null)
                }
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "detectBestFaceFromImage failed", e)
                callback(null)
            }
    }

    fun close() {
        detector.close()
        Log.d(TAG, "ML Kit FaceDetector closed")
    }
}
