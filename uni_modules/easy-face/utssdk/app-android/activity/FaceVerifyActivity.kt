package uts.sdk.modules.easyFace.face

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.*
import android.graphics.drawable.GradientDrawable
import android.hardware.camera2.*
import android.media.Image
import android.media.ImageReader
import android.os.*
import android.util.Log
import android.view.*
import android.widget.FrameLayout
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

/**
 * 原生人脸验证 Activity
 *
 * 完全原生实现：Camera2 预览 + 自定义 Canvas 覆盖层 + AI 管线
 * 从 uni-app 通过 startActivityForResult 启动，验证结果通过 Intent extras 返回
 *
 * 返回 Intent extras:
 *   "code": Int       - 结果码 (1=通过, 其他=失败/取消)
 *   "msg": String     - 结果描述
 *   "matchCount": Int - 匹配次数
 */
class FaceVerifyActivity : Activity() {

    companion object {
        private const val TAG = "FaceVerifyAct"
        private const val REQUEST_CAMERA_PERMISSION = 2001

        // Intent extra keys
        const val RESULT_CODE = "code"
        const val RESULT_MSG = "msg"
        const val RESULT_MATCH_COUNT = "matchCount"

        private const val CAPTURE_WIDTH = 640
        private const val CAPTURE_HEIGHT = 480
        private const val MATCH_MIN = 2
        private const val MAX_FAILS = 20
        private const val TIMEOUT_MS = 15000L
    }

    // Camera2
    private var cameraManager: CameraManager? = null
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null
    private var bgThread: HandlerThread? = null
    private var bgHandler: Handler? = null
    @Volatile private var isRunning = false
    @Volatile private var isProcessing = false
    private var sensorOrientation = 270 // 前置摄像头默认 270°

    // YUV→Bitmap 转换可复用缓冲区，避免每帧重新分配 + 消除逐像素 JNI 调用
    private var yBytes = ByteArray(0)
    private var uBytes = ByteArray(0)
    private var vBytes = ByteArray(0)
    private var argbPixels = IntArray(0)

    // Views
    private var textureView: TextureView? = null
    private var overlayView: OverlayView? = null

    // AI pipeline
    private var faceDetector: FaceDetectorHelper? = null
    private var faceNet: FaceNetHelper? = null
    private var livenessHelper: LivenessHelper? = null
    private var faceRepo: FaceRepository? = null

    // Parameters
    private var faceID = ""
    private var threshold = 0.6f
    private var needLiveness = true
    private var frontCameraId = ""

    // State
    @Volatile private var matchCount = 0
    @Volatile private var failCount = 0
    private var timeoutHandler: Handler? = null
    private var timeoutRunnable: Runnable? = null
    private var isVerifying = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Full screen
        window.addFlags(WindowManager.LayoutParams.FLAG_TRANSLUCENT_STATUS)
        window.addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN)
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            window.attributes.layoutInDisplayCutoutMode =
                WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
        }

        faceID = intent.getStringExtra("faceID") ?: ""
        threshold = intent.getDoubleExtra("threshold", 0.6).toFloat().coerceIn(0.5f, 0.95f)
        needLiveness = intent.getBooleanExtra("needLiveness", true)

        if (faceID.isEmpty()) {
            finishWithError(9010009, "faceID 不能为空")
            return
        }

        // Check camera permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                REQUEST_CAMERA_PERMISSION
            )
        } else {
            initAndStart()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                initAndStart()
            } else {
                finishWithError(9010001, "相机权限未授权")
            }
        }
    }

    private fun initAndStart() {
        // Init AI pipeline
        faceDetector = FaceDetectorHelper()
        faceNet = FaceNetHelper(this)
        livenessHelper = LivenessHelper(this)
        faceRepo = FaceRepository(this)

        // Check registered features
        if (faceRepo!!.getFeature(faceID) == null) {
            finishWithError(6, "未找到已注册的人脸特征，请先录入")
            return
        }

        // Create TextureView + Overlay
        val rootLayout = FrameLayout(this)
        setContentView(rootLayout)

        val w = resources.displayMetrics.widthPixels.toFloat()
        val h = resources.displayMetrics.heightPixels.toFloat()
        // 3:4 竖屏比例：采集 640x480 横屏 → sensorOrientation=270° 旋转后 480x640 竖屏
        val previewWidth = minOf(w * 0.75f, h * 0.55f * 3f / 4f)
        val previewHeight = previewWidth * 4f / 3f

        textureView = TextureView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                previewWidth.toInt(),
                previewHeight.toInt()
            ).apply {
                leftMargin = ((w - previewWidth) / 2f).toInt()
                topMargin = (h * 0.12f).toInt()
            }
            surfaceTextureListener = surfaceListener
        }
        rootLayout.addView(textureView)

        overlayView = OverlayView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
            onStartClick = { startVerify() }
            onBackClick = { goBack() }
        }
        rootLayout.addView(overlayView)
    }

    private val surfaceListener = object : TextureView.SurfaceTextureListener {
        override fun onSurfaceTextureAvailable(
            surface: SurfaceTexture,
            width: Int,
            height: Int
        ) {
            openCamera()
        }

        override fun onSurfaceTextureSizeChanged(
            surface: SurfaceTexture,
            width: Int,
            height: Int
        ) {}

        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
            return true
        }

        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {}
    }

    private fun openCamera() {
        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        frontCameraId = findFrontCameraId()
        if (frontCameraId.isEmpty()) {
            finishWithError(9010001, "未找到前置摄像头")
            return
        }

        bgThread = HandlerThread("FaceVerifyCam").apply {
            priority = Thread.MAX_PRIORITY
            start()
        }
        bgHandler = Handler(bgThread!!.looper)

        imageReader = ImageReader.newInstance(
            CAPTURE_WIDTH, CAPTURE_HEIGHT,
            ImageFormat.YUV_420_888, 2
        ).apply {
            setOnImageAvailableListener({ reader ->
                if (!isRunning) return@setOnImageAvailableListener
                try {
                    val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                    if (!isProcessing) {
                        isProcessing = true
                        // 注意：Image 由 processYuvFrame 回调负责关闭
                        processYuvFrame(image)
                    } else {
                        image.close()
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "acquireLatestImage error", e)
                    isProcessing = false
                }
            }, bgHandler)
        }

        try {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED
            ) return

            cameraManager!!.openCamera(frontCameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(device: CameraDevice) {
                    createSession(device)
                }

                override fun onDisconnected(device: CameraDevice) {
                    device.close()
                }

                override fun onError(device: CameraDevice, error: Int) {
                    device.close()
                    runOnUiThread {
                        Toast.makeText(
                            this@FaceVerifyActivity,
                            "摄像头错误",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }
            }, bgHandler)
        } catch (e: Exception) {
            Log.e(TAG, "openCamera failed", e)
            finishWithError(9019999, "打开摄像头失败: ${e.message}")
        }
    }

    private fun createSession(device: CameraDevice) {
        cameraDevice = device
        isRunning = true

        try {
            val surfaces = mutableListOf<Surface>(imageReader!!.surface)
            val textureSurface = Surface(textureView!!.surfaceTexture)
            surfaces.add(textureSurface)

            device.createCaptureSession(surfaces, object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    captureSession = session
                    try {
                        val builder = device.createCaptureRequest(
                            CameraDevice.TEMPLATE_PREVIEW
                        ).apply {
                            addTarget(imageReader!!.surface)
                            addTarget(textureSurface)
                            set(
                                CaptureRequest.CONTROL_AF_MODE,
                                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
                            )
                            set(
                                CaptureRequest.CONTROL_AE_MODE,
                                CaptureRequest.CONTROL_AE_MODE_ON
                            )
                            set(
                                CaptureRequest.CONTROL_AWB_MODE,
                                CaptureRequest.CONTROL_AWB_MODE_AUTO
                            )
                        }
                        session.setRepeatingRequest(builder.build(), null, bgHandler)
                    } catch (e: Exception) {
                        Log.e(TAG, "setRepeatingRequest failed", e)
                        finishWithError(9019999, "相机预览失败")
                    }
                }

                override fun onConfigureFailed(session: CameraCaptureSession) {
                    finishWithError(9010001, "摄像头配置失败")
                }
            }, bgHandler)
        } catch (e: Exception) {
            Log.e(TAG, "createCaptureSession failed", e)
            finishWithError(9019999, "创建会话失败: ${e.message}")
        }
    }

    // ─── High-Performance Frame Processing ───────────

    /**
     * 计算传给 ML Kit 的旋转角度
     * 前置摄像头需要特殊处理：镜像 + sensor 方向补偿
     */
    private fun getDetectorRotation(): Int {
        val deviceRotation = windowManager.defaultDisplay.rotation
        var degrees = 0
        when (deviceRotation) {
            Surface.ROTATION_0 -> degrees = 0
            Surface.ROTATION_90 -> degrees = 90
            Surface.ROTATION_180 -> degrees = 180
            Surface.ROTATION_270 -> degrees = 270
        }
        // 前摄像头：补偿 sensor orientation 并镜像
        val rotation = (sensorOrientation + degrees) % 360
        val mirrored = (360 - rotation) % 360
        return when (mirrored) {
            0 -> 0
            90 -> 90
            180 -> 180
            270 -> 270
            else -> 0
        }
    }

    /**
     * 处理 YUV 帧：先转为 Bitmap 并旋转到竖屏（与注册人脸方向一致），
     * 然后在竖屏图像上检测→裁剪→提取特征。
     * 这确保了验证和注册环节的人脸坐标系完全一致，提高特征匹配率。
     */
    private fun processYuvFrame(image: Image) {
        if (!isVerifying) {
            isProcessing = false
            image.close()
            return
        }

        var t0 = SystemClock.elapsedRealtime()

        // 步骤1：YUV → 原始横屏 Bitmap，立即释放 Image 缓冲区
        val rawBitmap = yuvToBitmap(image)
        image.close()

        if (rawBitmap == null) {
            isProcessing = false
            return
        }

        // 步骤2：旋转全帧到竖屏（与注册时的系统相机输出方向一致）
        val uprightBitmap = if (sensorOrientation != 0) {
            val matrix = Matrix()
            matrix.postRotate(sensorOrientation.toFloat())
            val rotated = Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true)
            rawBitmap.recycle()
            rotated
        } else {
            rawBitmap
        }
        Log.d(TAG, "YUV→upright Bitmap 耗时: ${SystemClock.elapsedRealtime() - t0}ms, size=${uprightBitmap.width}x${uprightBitmap.height}")

        // 步骤3：在竖屏图像上检测人脸（同步，后台线程）
        t0 = SystemClock.elapsedRealtime()
        val faceMap = faceDetector!!.detectBestFace(uprightBitmap)
        Log.d(TAG, "detectBestFace(Bitmap) 耗时: ${SystemClock.elapsedRealtime() - t0}ms")

        if (faceMap == null) {
            uprightBitmap.recycle()
            isProcessing = false
            return
        }

        val bbox = RectF(
            faceMap["left"] as Float,
            faceMap["top"] as Float,
            faceMap["right"] as Float,
            faceMap["bottom"] as Float
        )
        val leftEye = PointF(
            faceMap["leftEyeX"] as Float,
            faceMap["leftEyeY"] as Float
        )
        val rightEye = PointF(
            faceMap["rightEyeX"] as Float,
            faceMap["rightEyeY"] as Float
        )

        // 步骤4：从竖屏图像裁剪人脸（bbox 已在竖屏坐标系）
        val faceBitmap = cropFaceFromUprightBitmap(uprightBitmap, bbox)
        uprightBitmap.recycle()

        if (faceBitmap == null) {
            Log.w(TAG, "cropFace returned null")
            isProcessing = false
            return
        }

        try {
            // 活体检测
            if (needLiveness && livenessHelper!!.isAvailable()) {
                val cropBbox = RectF(0f, 0f, faceBitmap.width.toFloat(), faceBitmap.height.toFloat())
                val liveness = livenessHelper!!.detectLiveness(faceBitmap, cropBbox)
                if (liveness >= 0f && liveness < 0.65f) {
                    faceBitmap.recycle()
                    isProcessing = false
                    return
                }
            }

            // FaceNet 特征提取（眼坐标已在竖屏坐标系，与注册时一致）
            t0 = SystemClock.elapsedRealtime()
            val mappedLeftEye = mapPointToCrop(leftEye, bbox, faceBitmap.width, faceBitmap.height)
            val mappedRightEye = mapPointToCrop(rightEye, bbox, faceBitmap.width, faceBitmap.height)
            val fullBbox = RectF(0f, 0f, faceBitmap.width.toFloat(), faceBitmap.height.toFloat())
            val embedding = faceNet!!.extractEmbedding(faceBitmap, fullBbox, mappedLeftEye, mappedRightEye)
            Log.d(TAG, "extractEmbedding 耗时: ${SystemClock.elapsedRealtime() - t0}ms")

            if (embedding == null) {
                faceBitmap.recycle()
                isProcessing = false
                return
            }

            val stored = faceRepo!!.getFeature(faceID)
            if (stored == null) {
                faceBitmap.recycle()
                isProcessing = false
                return
            }

            t0 = SystemClock.elapsedRealtime()
            val similarity = faceRepo!!.cosineSimilarity(embedding, stored)
            Log.d(TAG, "cosineSimilarity 耗时: ${SystemClock.elapsedRealtime() - t0}ms,similarity: $similarity")
            faceBitmap.recycle()

            runOnUiThread {
                if (similarity >= threshold) {
                    matchCount++
                    failCount = 0
                    overlayView?.updateMatchCount(matchCount, failCount)
                    if (matchCount >= MATCH_MIN) {
                        onVerifySuccess()
                    }
                } else {
                    failCount++
                    matchCount = maxOf(0, matchCount - 1)
                    overlayView?.updateMatchCount(matchCount, failCount)
                    if (failCount >= MAX_FAILS) {
                        onVerifyFailed()
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "processYuvFrame pipeline error", e)
            try { faceBitmap.recycle() } catch (_: Exception) {}
        } finally {
            isProcessing = false
        }
    }

    /**
     * 将点从原始图像坐标系映射到裁剪 Bitmap 坐标系
     */
    private fun mapPointToCrop(point: PointF, bbox: RectF, cropW: Int, cropH: Int): PointF {
        return PointF(
            ((point.x - bbox.left) / bbox.width() * cropW).coerceIn(0f, cropW.toFloat()),
            ((point.y - bbox.top) / bbox.height() * cropH).coerceIn(0f, cropH.toFloat())
        )
    }

    /**
     * 从竖屏 Bitmap 裁剪人脸区域（bbox 已在竖屏坐标系中，无需旋转）
     * 扩展 15% margin，保证 FaceNet 有足够上下文
     */
    private fun cropFaceFromUprightBitmap(uprightBitmap: Bitmap, bbox: RectF): Bitmap? {
        try {
            val marginX = bbox.width() * 0.15f
            val marginY = bbox.height() * 0.15f
            val cropLeft = ((bbox.left - marginX).coerceAtLeast(0f)).toInt()
            val cropTop = ((bbox.top - marginY * 1.2f).coerceAtLeast(0f)).toInt()
            val cropRight = ((bbox.right + marginX).coerceAtMost(uprightBitmap.width.toFloat())).toInt()
            val cropBottom = ((bbox.bottom + marginY * 0.5f).coerceAtMost(uprightBitmap.height.toFloat())).toInt()

            val cropW = cropRight - cropLeft
            val cropH = cropBottom - cropTop
            if (cropW <= 10 || cropH <= 10) return null

            return Bitmap.createBitmap(uprightBitmap, cropLeft, cropTop, cropW, cropH)
        } catch (e: Exception) {
            Log.e(TAG, "cropFace error", e)
            return null
        }
    }

    /**
     * 高效 YUV_420_888 → Bitmap 转换（无 JPEG 压缩）
     *
     * 优化策略：
     * 1. 批量拷贝 Y/U/V plane 到 Java byte[] — 仅 3 次 JNI 调用，而非 921,600 次逐像素调用
     * 2. 复用 byte[]/int[] 缓冲区，避免每帧重新分配 ~2.4 MB
     * 3. 纯 Java 侧的 YUV→ARGB 转换在 CPU cache 中高效执行
     */
    private fun yuvToBitmap(image: Image): Bitmap? {
        try {
            val w = image.width
            val h = image.height
            val planes = image.planes
            if (planes.size < 3) return null

            val yPlane = planes[0]
            val uPlane = planes[1]
            val vPlane = planes[2]

            val yBuffer = yPlane.buffer
            val uBuffer = uPlane.buffer
            val vBuffer = vPlane.buffer

            val yRowStride = yPlane.rowStride
            val uRowStride = uPlane.rowStride
            val vRowStride = vPlane.rowStride
            val uPixelStride = uPlane.pixelStride
            val vPixelStride = vPlane.pixelStride

            // 不能简单用 rowStride * height，因为最后一行末尾没有填充字节
            // 会导致 BufferUnderflowException，必须以 buffer.remaining() 为准
            val ySize = minOf(yRowStride * h, yBuffer.remaining())
            val uSize = minOf(uRowStride * h / 2, uBuffer.remaining())
            val vSize = minOf(vRowStride * h / 2, vBuffer.remaining())
            val pixelCount = w * h

            // 复用或按需扩容缓冲区（单线程保证，由 isProcessing 原子锁保护）
            if (yBytes.size < ySize) yBytes = ByteArray(ySize)
            if (uBytes.size < uSize) uBytes = ByteArray(uSize)
            if (vBytes.size < vSize) vBytes = ByteArray(vSize)
            if (argbPixels.size < pixelCount) argbPixels = IntArray(pixelCount)

            // ★ 关键优化：3 次批量 JNI 拷贝替代 921,600 次逐像素 JNI 调用
            yBuffer.get(yBytes, 0, ySize)
            uBuffer.get(uBytes, 0, uSize)
            vBuffer.get(vBytes, 0, vSize)

            // 纯 Java 侧 YUV→ARGB，无 JNI 开销
            val rowStrideGap = yRowStride - w
            var yIdx = 0

            for (row in 0 until h) {
                val uvRow = row shr 1
                val uvRowBaseU = uvRow * uRowStride
                val uvRowBaseV = uvRow * vRowStride

                for (col in 0 until w) {
                    val y = yBytes[yIdx].toInt() and 0xFF
                    yIdx++

                    val uvCol = col shr 1
                    val uIdx = uvRowBaseU + uvCol * uPixelStride
                    val vIdx = uvRowBaseV + uvCol * vPixelStride

                    val u = (uBytes[uIdx.coerceIn(0, uSize - 1)].toInt() and 0xFF) - 128
                    val v = (vBytes[vIdx.coerceIn(0, vSize - 1)].toInt() and 0xFF) - 128

                    // YUV → RGB (BT.601)
                    val r = (y + (v * 1436 shr 10)).coerceIn(0, 255)
                    val g = (y - (u * 352 + v * 731 shr 10)).coerceIn(0, 255)
                    val b = (y + (u * 1815 shr 10)).coerceIn(0, 255)

                    argbPixels[row * w + col] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
                // 跳过 Y 行间填充
                yIdx += rowStrideGap
            }

            return Bitmap.createBitmap(argbPixels, w, h, Bitmap.Config.ARGB_8888)
        } catch (e: Exception) {
            Log.e(TAG, "yuvToBitmap error", e)
            return null
        }
    }

    private fun startVerify() {
        if (isVerifying) return
        isVerifying = true
        matchCount = 0
        failCount = 0
        overlayView?.startVerifying()

        // Timeout
        timeoutHandler = Handler(Looper.getMainLooper())
        timeoutRunnable = Runnable {
            if (isVerifying) {
                isVerifying = false
                stopCamera()
                finishWithResult(9010004, "验证超时，请重试", matchCount)
            }
        }
        timeoutHandler?.postDelayed(timeoutRunnable!!, TIMEOUT_MS)
    }

    private fun onVerifySuccess() {
        if (!isVerifying) return
        isVerifying = false
        cancelTimeout()
        stopCamera()
        runOnUiThread {
            Toast.makeText(this, "人脸识别成功", Toast.LENGTH_SHORT).show()
            Handler(Looper.getMainLooper()).postDelayed({
                finishWithResult(1, "人脸验证通过", matchCount)
            }, 400)
        }
    }

    private fun onVerifyFailed() {
        if (!isVerifying) return
        isVerifying = false
        cancelTimeout()
        stopCamera()
        runOnUiThread {
            finishWithResult(9010005, "验证失败次数过多", matchCount)
        }
    }

    private fun goBack() {
        isVerifying = false
        cancelTimeout()
        stopCamera()
        finishWithResult(0, "用户取消", matchCount)
    }

    private fun finishWithError(code: Int, msg: String) {
        destroyAll()
        val intent = Intent().apply {
            putExtra(RESULT_CODE, code)
            putExtra(RESULT_MSG, msg)
            putExtra(RESULT_MATCH_COUNT, 0)
        }
        setResult(Activity.RESULT_OK, intent)
        finish()
    }

    private fun finishWithResult(code: Int, msg: String, matchCount: Int) {
        destroyAll()
        val intent = Intent().apply {
            putExtra(RESULT_CODE, code)
            putExtra(RESULT_MSG, msg)
            putExtra(RESULT_MATCH_COUNT, matchCount)
        }
        setResult(Activity.RESULT_OK, intent)
        finish()
    }

    private fun cancelTimeout() {
        timeoutHandler?.removeCallbacks(timeoutRunnable ?: return)
    }

    // ─── Camera helpers ──────────────────────────────

    private fun findFrontCameraId(): String {
        try {
            for (id in cameraManager!!.cameraIdList) {
                val characteristics = cameraManager!!.getCameraCharacteristics(id)
                val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    val orientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION)
                    if (orientation != null) {
                        sensorOrientation = orientation
                    }
                    return id
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "findFrontCameraId error", e)
        }
        return ""
    }

    private fun stopCamera() {
        isRunning = false
        try {
            captureSession?.close()
        } catch (_: Exception) {}
        try {
            cameraDevice?.close()
        } catch (_: Exception) {}
        try {
            imageReader?.close()
        } catch (_: Exception) {}
        try {
            bgThread?.quitSafely()
        } catch (_: Exception) {}
        captureSession = null
        cameraDevice = null
        imageReader = null
        bgHandler = null
        bgThread = null
    }

    /**
     * 释放 AI 管线持有的 native 资源（TFLite Interpreter / ML Kit detector）
     * 必须在 Activity 销毁前调用，否则 native 内存泄漏导致后续启动性能急剧下降
     */
    private fun releaseAiHelpers() {
        try { faceDetector?.close() } catch (_: Exception) {}
        try { faceNet?.close() } catch (_: Exception) {}
        try { livenessHelper?.close() } catch (_: Exception) {}
        faceDetector = null
        faceNet = null
        livenessHelper = null
    }

    /**
     * 统一清理：相机 + AI 管线 + UI 动画
     */
    private fun destroyAll() {
        isVerifying = false
        isProcessing = false
        cancelTimeout()
        overlayView?.stopAnimation()
        stopCamera()
        releaseAiHelpers()
    }

    // ─── Lifecycle ───────────────────────────────────

    override fun onResume() {
        super.onResume()
        if (!isRunning && textureView?.isAvailable == true) {
            openCamera()
        }
    }

    override fun onPause() {
        isVerifying = false
        cancelTimeout()
        stopCamera()
        super.onPause()
    }

    override fun onDestroy() {
        destroyAll()
        super.onDestroy()
    }

    override fun onBackPressed() {
        goBack()
    }

    // ═══════════════════════════════════════════════════
    // Custom Overlay View
    // ═══════════════════════════════════════════════════

    inner class OverlayView(context: Context) : View(context) {

        var onStartClick: (() -> Unit)? = null
        var onBackClick: (() -> Unit)? = null

        private var matchCount = 0
        private var failCount = 0
        private var isVerifyingLocal = false
        private var scanAnimPhase = 0f
        private val scanAnimRunnable = object : Runnable {
            override fun run() {
                if (isVerifyingLocal) {
                    scanAnimPhase += 0.05f
                    if (scanAnimPhase > Math.PI * 2) scanAnimPhase = 0f
                    invalidate()
                    postDelayed(this, 16)
                }
            }
        }

        // Paints
        private val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(40, 0, 0, 0)
            style = Paint.Style.FILL
        }
        private val cornerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(153, 255, 255, 255) // 0.6 alpha
            style = Paint.Style.STROKE
            strokeWidth = 4f.dp
        }
        private val scanPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(153, 107, 76, 230)
            style = Paint.Style.FILL
        }
        private val dotActivePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(34, 197, 94)
            style = Paint.Style.FILL
        }
        private val dotInactivePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(38, 255, 255, 255)
            style = Paint.Style.FILL
        }
        private val titlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 22f.sp
            isFakeBoldText = true
            textAlign = Paint.Align.CENTER
            typeface = Typeface.DEFAULT_BOLD
        }
        private val subtitlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(102, 255, 255, 255) // 0.4 alpha
            textSize = 13f.sp
            textAlign = Paint.Align.CENTER
        }
        private val progressTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(102, 255, 255, 255)
            textSize = 12f.sp
            textAlign = Paint.Align.CENTER
        }
        private val overlayTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(179, 255, 255, 255) // 0.7 alpha
            textSize = 12f.sp
            textAlign = Paint.Align.CENTER
        }
        private val pulseRingPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.TRANSPARENT
            style = Paint.Style.STROKE
            strokeWidth = 3f.dp
        }.apply {
            color = Color.argb(153, 107, 76, 230)
        }
        private val btnBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { isFilterBitmap = true }
        private val btnTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 15f.sp
            isFakeBoldText = true
            textAlign = Paint.Align.CENTER
            typeface = Typeface.DEFAULT_BOLD
        }
        private val btnBackBorderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(26, 255, 255, 255)
            style = Paint.Style.STROKE
            strokeWidth = 2f.dp
        }
        private val btnBackTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(179, 255, 255, 255)
            textSize = 14f.sp
            textAlign = Paint.Align.CENTER
        }

        // Button hit rects
        private var btnPrimaryRect = RectF()
        private var btnBackRect = RectF()

        // Button gradients
        private val btnGradient = GradientDrawable(
            GradientDrawable.Orientation.LEFT_RIGHT,
            intArrayOf(Color.rgb(74, 108, 247), Color.rgb(107, 76, 230))
        )
        private val btnDisabledGradient = GradientDrawable(
            GradientDrawable.Orientation.LEFT_RIGHT,
            intArrayOf(
                Color.argb(128, 74, 108, 247),
                Color.argb(128, 107, 76, 230)
            )
        )

        fun startVerifying() {
            isVerifyingLocal = true
            matchCount = 0
            failCount = 0
            scanAnimPhase = 0f
            removeCallbacks(scanAnimRunnable)
            post(scanAnimRunnable)
            invalidate()
        }

        fun stopAnimation() {
            isVerifyingLocal = false
            removeCallbacks(scanAnimRunnable)
        }

        fun updateMatchCount(matched: Int, failed: Int) {
            matchCount = matched
            failCount = failed
            invalidate()
        }

        override fun onTouchEvent(event: MotionEvent): Boolean {
            if (event.action == MotionEvent.ACTION_UP) {
                val x = event.x
                val y = event.y
                if (btnPrimaryRect.contains(x, y) && !isVerifyingLocal) {
                    onStartClick?.invoke()
                    return true
                }
                if (btnBackRect.contains(x, y)) {
                    onBackClick?.invoke()
                    return true
                }
            }
            return true
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val w = width.toFloat()
            val h = height.toFloat()

            // ── Camera frame area (3:4 竖屏，与旋转后 480x640 一致) ──
            val frameWidth = minOf(w * 0.75f, h * 0.55f * 3f / 4f)
            val frameHeight = frameWidth * 4f / 3f
            val frameLeft = (w - frameWidth) / 2f
            val frameTop = h * 0.12f
            val frameRight = frameLeft + frameWidth
            val frameBottom = frameTop + frameHeight
            val cornerLen = minOf(frameWidth, frameHeight) * 0.08f
            val cornerStroke = cornerPaint.strokeWidth

            // Top-left corner
            canvas.drawLine(
                frameLeft, frameTop + cornerLen,
                frameLeft, frameTop + cornerStroke / 2, cornerPaint
            )
            canvas.drawLine(
                frameLeft, frameTop + cornerStroke / 2,
                frameLeft + cornerLen, frameTop + cornerStroke / 2, cornerPaint
            )

            // Top-right corner
            canvas.drawLine(
                frameRight - cornerLen, frameTop + cornerStroke / 2,
                frameRight, frameTop + cornerStroke / 2, cornerPaint
            )
            canvas.drawLine(
                frameRight, frameTop + cornerStroke / 2,
                frameRight, frameTop + cornerLen, cornerPaint
            )

            // Bottom-left corner
            canvas.drawLine(
                frameLeft, frameBottom - cornerLen,
                frameLeft, frameBottom - cornerStroke / 2, cornerPaint
            )
            canvas.drawLine(
                frameLeft, frameBottom - cornerStroke / 2,
                frameLeft + cornerLen, frameBottom - cornerStroke / 2, cornerPaint
            )

            // Bottom-right corner
            canvas.drawLine(
                frameRight - cornerLen, frameBottom - cornerStroke / 2,
                frameRight, frameBottom - cornerStroke / 2, cornerPaint
            )
            canvas.drawLine(
                frameRight, frameBottom - cornerStroke / 2,
                frameRight, frameBottom - cornerLen, cornerPaint
            )

            // ── Scan line (when verifying) ──────────
            if (isVerifyingLocal) {
                val scanProgress = (Math.sin(scanAnimPhase.toDouble()) + 1.0) / 2.0  // 0..1
                val scanY = frameTop + (frameHeight * 0.1f) + (frameHeight * 0.8f * scanProgress.toFloat())
                canvas.drawRect(frameLeft, scanY, frameRight, scanY + 3f.dp, scanPaint)
            }

            // ── Title ────────────────────────────────
            canvas.drawText("人脸验证", w / 2f, h * 0.05f, titlePaint)
            val statusText = when {
                failCount >= MAX_FAILS -> "验证失败次数过多"
                isVerifyingLocal -> "正在进行人脸验证..."
                else -> "请将脸部对准摄像头进行验证"
            }
            canvas.drawText(statusText, w / 2f, h * 0.05f + 24f.dp, subtitlePaint)

            // ── Progress dots ────────────────────────
            val dotCount = 5
            val dotRadius = 7f.dp
            val dotSpacing = 22f.dp
            val dotsTotalWidth = dotCount * (dotRadius * 2) + (dotCount - 1) * dotSpacing
            val dotsStartX = (w - dotsTotalWidth) / 2f + dotRadius
            val dotsY = frameBottom + 24f.dp
            for (i in 0 until dotCount) {
                val cx = dotsStartX + i * (dotRadius * 2 + dotSpacing)
                canvas.drawCircle(cx, dotsY, dotRadius, if (i < matchCount) dotActivePaint else dotInactivePaint)
            }

            // Progress text
            val progressText = if (isVerifyingLocal) "请正对摄像头，保持面部清晰" else "点击按钮开始人脸验证"
            canvas.drawText(progressText, w / 2f, dotsY + dotRadius + 18f.dp, progressTextPaint)

            // ── Overlay (when verifying) ─────────────
            if (isVerifyingLocal) {
                canvas.drawRect(frameLeft, frameTop, frameRight, frameBottom, bgPaint)
            }

            // ── Buttons ──────────────────────────────
            val btnWidth = w * 0.78f
            val btnHeight = 42f.dp
            val btnStartX = (w - btnWidth) / 2f
            val primaryBtnY = dotsY + dotRadius + 38f.dp
            val backBtnY = primaryBtnY + btnHeight + 12f.dp

            btnPrimaryRect.set(btnStartX, primaryBtnY, btnStartX + btnWidth, primaryBtnY + btnHeight)

            // Primary button
            val gradient = if (isVerifyingLocal || faceID.isEmpty()) btnDisabledGradient else btnGradient
            gradient.setBounds(
                btnStartX.toInt(),
                primaryBtnY.toInt(),
                (btnStartX + btnWidth).toInt(),
                (primaryBtnY + btnHeight).toInt()
            )
            gradient.cornerRadius = 12f.dp
            gradient.draw(canvas)

            val btnText = if (isVerifyingLocal) "正在验证..." else "开始人脸验证"
            if (isVerifyingLocal) btnTextPaint.alpha = 128 else btnTextPaint.alpha = 255
            canvas.drawText(
                btnText,
                w / 2f,
                primaryBtnY + btnHeight / 2f - (btnTextPaint.descent() + btnTextPaint.ascent()) / 2f,
                btnTextPaint
            )

            // Back button
            btnBackRect.set(btnStartX, backBtnY, btnStartX + btnWidth, backBtnY + btnHeight)
            val backGradient = GradientDrawable()
            backGradient.setColor(Color.TRANSPARENT)
            backGradient.cornerRadius = 10f.dp
            backGradient.setStroke(2f.dp.toInt(), Color.argb(26, 255, 255, 255))
            backGradient.setBounds(
                btnStartX.toInt(),
                backBtnY.toInt(),
                (btnStartX + btnWidth).toInt(),
                (backBtnY + btnHeight).toInt()
            )
            backGradient.draw(canvas)
            canvas.drawText(
                "返回",
                w / 2f,
                backBtnY + btnHeight / 2f - (btnBackTextPaint.descent() + btnBackTextPaint.ascent()) / 2f,
                btnBackTextPaint
            )

            // ── Warning ──────────────────────────────
            if (failCount >= MAX_FAILS) {
                val warnY = backBtnY + btnHeight + 20f.dp
                canvas.drawText(
                    "验证失败次数过多，请返回使用账号密码登录",
                    w / 2f, warnY, subtitlePaint
                )
            }
        }
    }

    // ─── Extension properties for dp/sp conversion ───

    private val Float.dp: Float get() = this * resources.displayMetrics.density
    private val Float.sp: Float get() = this * resources.displayMetrics.scaledDensity
}
