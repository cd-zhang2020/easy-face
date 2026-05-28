package uts.sdk.modules.easyFace.face

import android.Manifest
import android.animation.*
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.*
import android.graphics.drawable.GradientDrawable
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.media.Image
import android.os.*
import android.util.Log
import android.view.*
import android.view.animation.*
import android.widget.FrameLayout
import android.widget.Toast
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

/**
 * CameraX 版本的人脸验证 Activity
 *
 * 使用 androidx.camera (CameraX) 替代 Camera2 API：
 * - PreviewView 自动管理 SurfaceTexture 生命周期
 * - ImageAnalysis 在后台线程异步分发帧
 * - ProcessCameraProvider 自动处理摄像头绑定与生命周期
 *
 * UI 特点：
 * - 深色渐变动态背景
 * - 椭圆/圆角人脸框 + 发光角标
 * - 扫描线平滑动画 (sin 缓动)
 * - 脉冲光环呼吸动画
 * - 验证进度指示点 (带脉冲效果)
 * - 成功粒子爆炸动画
 * - 毛玻璃风格按钮
 *
 * 返回 Intent extras:
 *   "code": Int       - 结果码 (1=通过, 其他=失败/取消)
 *   "msg": String     - 结果描述
 *   "matchCount": Int - 匹配次数
 */
class CameraXFaceVerifyActivity : Activity(), LifecycleOwner {

    companion object {
        private const val TAG = "CameraXFaceVerify"
        private const val REQUEST_CAMERA_PERMISSION = 2002

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

    // Lifecycle
    private val lifecycleRegistry = LifecycleRegistry(this)
    override fun getLifecycle(): Lifecycle = lifecycleRegistry

    // CameraX
    private var cameraProvider: ProcessCameraProvider? = null
    private var camera: androidx.camera.core.Camera? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var cameraExecutor = Executors.newSingleThreadExecutor()

    @Volatile private var isVerifying = false
    @Volatile private var isProcessing = false
    private var sensorOrientation = 270 // 前置摄像头默认 270°

    // Views
    private var previewView: PreviewView? = null
    private var overlayView: CameraXOverlayView? = null

    // AI pipeline
    private var faceDetector: FaceDetectorHelper? = null
    private var faceNet: FaceNetHelper? = null
    private var livenessHelper: LivenessHelper? = null
    private var faceRepo: FaceRepository? = null

    // Parameters
    private var faceID = ""
    private var threshold = 0.6f
    private var needLiveness = true

    // State
    @Volatile private var matchCount = 0
    @Volatile private var failCount = 0
    private var timeoutHandler: Handler? = null
    private var timeoutRunnable: Runnable? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_CREATE)

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

        if (faceRepo!!.getFeature(faceID) == null) {
            finishWithError(6, "未找到已注册的人脸特征，请先录入")
            return
        }

        // Create layout
        val rootLayout = FrameLayout(this)
        setContentView(rootLayout)

        // PreviewView
        previewView = PreviewView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
            scaleType = PreviewView.ScaleType.FILL_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
        rootLayout.addView(previewView)

        // Overlay
        overlayView = CameraXOverlayView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
            onStartClick = { startVerify() }
            onBackClick = { goBack() }
        }
        rootLayout.addView(overlayView)

        // Start CameraX
        startCameraX()
    }

    private fun startCameraX() {
        // Get sensor orientation via CameraManager using ApplicationContext to avoid Activity leak
        val cm = applicationContext.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try {
            for (id in cm.cameraIdList) {
                val characteristics = cm.getCameraCharacteristics(id)
                val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    val orientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION)
                    if (orientation != null) {
                        sensorOrientation = orientation
                    }
                    break
                }
            }
        } catch (_: Exception) {}

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()

                // Find front camera
                val cameraSelector = CameraSelector.Builder()
                    .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                    .build()

                // Build Preview (不限制宽高比，让 CameraX 自适应屏幕，避免人脸被放大裁切)
                val preview = Preview.Builder()
                    .build()
                    .also {
                        it.setSurfaceProvider(previewView!!.surfaceProvider)
                    }

                // Build ImageAnalysis
                imageAnalysis = ImageAnalysis.Builder()
                    .setTargetResolution(android.util.Size(CAPTURE_HEIGHT, CAPTURE_WIDTH))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                    .build()
                    .also {
                        it.setAnalyzer(cameraExecutor, FrameAnalyzer())
                    }

                // Bind to lifecycle
                camera = cameraProvider!!.bindToLifecycle(
                    this@CameraXFaceVerifyActivity,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )
            } catch (e: Exception) {
                Log.e(TAG, "CameraX start failed", e)
                finishWithError(9019999, "启动摄像头失败: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // ═══════════════════════════════════════════════════
    // Frame Analyzer (ImageAnalysis)
    // ═══════════════════════════════════════════════════

    inner class FrameAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(imageProxy: ImageProxy) {
            if (!isVerifying) {
                imageProxy.close()
                return
            }
            if (isProcessing) {
                imageProxy.close()
                return
            }
            isProcessing = true
            try {
                val image: Image = imageProxy.image ?: run {
                    isProcessing = false
                    imageProxy.close()
                    return
                }
                val rotation = imageProxy.imageInfo.rotationDegrees
                processYuvFrame(image, rotation)
                imageProxy.close()
            } catch (e: Exception) {
                Log.e(TAG, "analyze error", e)
                isProcessing = false
                imageProxy.close()
            }
        }
    }

    private fun processYuvFrame(image: Image, rotationDegrees: Int) {
        var t0 = SystemClock.elapsedRealtime()

        // Step 1: YUV → raw Bitmap
        val rawBitmap = yuvToBitmap(image)
        if (rawBitmap == null) {
            isProcessing = false
            return
        }

        // Step 2: Rotate to upright (compensate sensor orientation)
        val uprightBitmap = if (sensorOrientation != 0) {
            val matrix = Matrix()
            matrix.postRotate(sensorOrientation.toFloat())
            val rotated =
                Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true)
            rawBitmap.recycle()
            rotated
        } else {
            rawBitmap
        }
        Log.d(
            TAG,
            "YUV→upright 耗时: ${SystemClock.elapsedRealtime() - t0}ms, size=${uprightBitmap.width}x${uprightBitmap.height}"
        )

        // Step 3: Detect face
        t0 = SystemClock.elapsedRealtime()
        val faceMap = faceDetector!!.detectBestFace(uprightBitmap)
        Log.d(TAG, "detectBestFace 耗时: ${SystemClock.elapsedRealtime() - t0}ms")

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

        // Step 4: Crop face
        val faceBitmap = cropFaceFromUprightBitmap(uprightBitmap, bbox)
        uprightBitmap.recycle()

        if (faceBitmap == null) {
            Log.w(TAG, "cropFace returned null")
            isProcessing = false
            return
        }

        try {
            // Liveness check
            if (needLiveness && livenessHelper!!.isAvailable()) {
                val cropBbox = RectF(0f, 0f, faceBitmap.width.toFloat(), faceBitmap.height.toFloat())
                val liveness = livenessHelper!!.detectLiveness(faceBitmap, cropBbox)
                if (liveness >= 0f && liveness < 0.65f) {
                    faceBitmap.recycle()
                    isProcessing = false
                    return
                }
            }

            // FaceNet embedding
            t0 = SystemClock.elapsedRealtime()
            val mappedLeftEye = mapPointToCrop(leftEye, bbox, faceBitmap.width, faceBitmap.height)
            val mappedRightEye = mapPointToCrop(rightEye, bbox, faceBitmap.width, faceBitmap.height)
            val fullBbox = RectF(0f, 0f, faceBitmap.width.toFloat(), faceBitmap.height.toFloat())
            val embedding =
                faceNet!!.extractEmbedding(faceBitmap, fullBbox, mappedLeftEye, mappedRightEye)
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
            Log.d(
                TAG,
                "cosineSimilarity 耗时: ${SystemClock.elapsedRealtime() - t0}ms, similarity: $similarity"
            )
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
            try {
                faceBitmap.recycle()
            } catch (_: Exception) {
            }
        } finally {
            isProcessing = false
        }
    }

    private fun mapPointToCrop(point: PointF, bbox: RectF, cropW: Int, cropH: Int): PointF {
        return PointF(
            ((point.x - bbox.left) / bbox.width() * cropW).coerceIn(0f, cropW.toFloat()),
            ((point.y - bbox.top) / bbox.height() * cropH).coerceIn(0f, cropH.toFloat())
        )
    }

    private fun cropFaceFromUprightBitmap(uprightBitmap: Bitmap, bbox: RectF): Bitmap? {
        try {
            val marginX = bbox.width() * 0.15f
            val marginY = bbox.height() * 0.15f
            val cropLeft = ((bbox.left - marginX).coerceAtLeast(0f)).toInt()
            val cropTop = ((bbox.top - marginY * 1.2f).coerceAtLeast(0f)).toInt()
            val cropRight =
                ((bbox.right + marginX).coerceAtMost(uprightBitmap.width.toFloat())).toInt()
            val cropBottom =
                ((bbox.bottom + marginY * 0.5f).coerceAtMost(uprightBitmap.height.toFloat())).toInt()

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
     * YUV_420_888 → ARGB_8888 Bitmap
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

            val ySize = minOf(yRowStride * h, yBuffer.remaining())
            val uSize = minOf(uRowStride * h / 2, uBuffer.remaining())
            val vSize = minOf(vRowStride * h / 2, vBuffer.remaining())

            val yBytes = ByteArray(ySize)
            val uBytes = ByteArray(uSize)
            val vBytes = ByteArray(vSize)

            yBuffer.get(yBytes)
            uBuffer.get(uBytes)
            vBuffer.get(vBytes)

            val pixels = IntArray(w * h)
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

                    val r = (y + (v * 1436 shr 10)).coerceIn(0, 255)
                    val g = (y - (u * 352 + v * 731 shr 10)).coerceIn(0, 255)
                    val b = (y + (u * 1815 shr 10)).coerceIn(0, 255)

                    pixels[row * w + col] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
                yIdx += rowStrideGap
            }

            return Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888)
        } catch (e: Exception) {
            Log.e(TAG, "yuvToBitmap error", e)
            return null
        }
    }

    // ═══════════════════════════════════════════════════
    // Verify lifecycle
    // ═══════════════════════════════════════════════════

    private fun startVerify() {
        if (isVerifying) return
        isVerifying = true
        matchCount = 0
        failCount = 0
        overlayView?.startVerifying()

        timeoutHandler = Handler(Looper.getMainLooper())
        timeoutRunnable = Runnable {
            if (isVerifying) {
                isVerifying = false
                overlayView?.stopAnimation()
                finishWithResult(9010004, "验证超时，请重试", matchCount)
            }
        }
        timeoutHandler?.postDelayed(timeoutRunnable!!, TIMEOUT_MS)
    }

    private fun onVerifySuccess() {
        if (!isVerifying) return
        isVerifying = false
        cancelTimeout()
        runOnUiThread {
            // Play success animation before closing
            overlayView?.onVerifySuccess()
            Handler(Looper.getMainLooper()).postDelayed({
                finishWithResult(1, "人脸验证通过", matchCount)
            }, 1200)
        }
    }

    private fun onVerifyFailed() {
        if (!isVerifying) return
        isVerifying = false
        cancelTimeout()
        runOnUiThread {
            overlayView?.onVerifyFailed()
            Handler(Looper.getMainLooper()).postDelayed({
                finishWithResult(9010005, "验证失败次数过多", matchCount)
            }, 800)
        }
    }

    private fun goBack() {
        isVerifying = false
        cancelTimeout()
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

    private fun releaseAiHelpers() {
        try {
            faceDetector?.close()
        } catch (_: Exception) {
        }
        try {
            faceNet?.close()
        } catch (_: Exception) {
        }
        try {
            livenessHelper?.close()
        } catch (_: Exception) {
        }
        faceDetector = null
        faceNet = null
        livenessHelper = null
    }

    private fun destroyAll() {
        isVerifying = false
        isProcessing = false
        cancelTimeout()
        overlayView?.stopAnimation()
        try {
            cameraExecutor.shutdownNow()
        } catch (_: Exception) {
        }
        try {
            cameraProvider?.unbindAll()
        } catch (_: Exception) {
        }
        cameraProvider = null
        camera = null
        imageAnalysis = null
        releaseAiHelpers()
    }

    // ═══════════════════════════════════════════════════
    // Lifecycle
    // ═══════════════════════════════════════════════════

    override fun onStart() {
        super.onStart()
        lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_START)
    }

    override fun onResume() {
        super.onResume()
        lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_RESUME)
    }

    override fun onPause() {
        super.onPause()
        lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_PAUSE)
    }

    override fun onStop() {
        super.onStop()
        lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_STOP)
    }

    override fun onDestroy() {
        lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_DESTROY)
        destroyAll()
        super.onDestroy()
    }

    override fun onBackPressed() {
        goBack()
    }

    // ═══════════════════════════════════════════════════
    // Custom Overlay View — beautiful animated UI
    // ═══════════════════════════════════════════════════

    inner class CameraXOverlayView(context: Context) : View(context) {

        var onStartClick: (() -> Unit)? = null
        var onBackClick: (() -> Unit)? = null

        // State
        private var matchCount = 0
        private var failCount = 0
        private var isVerifyingLocal = false
        private var showSuccess = false
        private var showFailure = false

        // ── Animation state ──
        private var scanPhase = 0f
        private var pulsePhase = 0f
        private var glowAlpha = 0f
        private var successPhase = 0f
        private var particles = mutableListOf<Particle>()
        private var alphaIn = 0f

        // ── Animators ──
        private var scanAnimator: ValueAnimator? = null
        private var pulseAnimator: ValueAnimator? = null
        private var successAnimator: ValueAnimator? = null
        private var particleAnimator: ValueAnimator? = null
        private var fadeInAnimator: ValueAnimator? = null

        // ── Button hit rects ──
        private var btnPrimaryRect = RectF()
        private var btnBackRect = RectF()

        // ── Gradient background ──
        private val bgGradient = LinearGradient(
            0f, 0f, 0f, 1f,
            intArrayOf(
                Color.rgb(10, 12, 30),
                Color.rgb(22, 20, 50),
                Color.rgb(12, 10, 35)
            ),
            floatArrayOf(0f, 0.5f, 1f),
            Shader.TileMode.CLAMP
        )

        // ── Paints ──

        // Corner bracket
        private val cornerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(220, 120, 200, 255)
            style = Paint.Style.STROKE
            strokeWidth = 3f.dp
            strokeCap = Paint.Cap.ROUND
        }
        private val cornerGlowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(80, 120, 200, 255)
            style = Paint.Style.STROKE
            strokeWidth = 8f.dp
            strokeCap = Paint.Cap.ROUND
        }

        // Pulsing ring
        private val pulseRingPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeCap = Paint.Cap.ROUND
        }

        // Scanning bar
        private val scanGradient = LinearGradient(
            0f, 0f, 0f, 1f,
            intArrayOf(
                Color.TRANSPARENT,
                Color.argb(100, 100, 180, 255),
                Color.argb(200, 130, 210, 255),
                Color.argb(100, 100, 180, 255),
                Color.TRANSPARENT
            ),
            floatArrayOf(0f, 0.3f, 0.5f, 0.7f, 1f),
            Shader.TileMode.CLAMP
        )
        private val scanPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
        }

        // Face frame background overlay (不透明遮罩，只露出椭圆预览区域)
        private val dimPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(255, 8, 10, 28)
            style = Paint.Style.FILL
        }

        // Progress dots
        private val dotActivePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(80, 210, 150)
            style = Paint.Style.FILL
        }
        private val dotActiveGlowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(60, 80, 210, 150)
            style = Paint.Style.FILL
        }
        private val dotInactivePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(50, 255, 255, 255)
            style = Paint.Style.FILL
        }
        private val dotFailedPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(255, 100, 100)
            style = Paint.Style.FILL
        }

        // Text
        private val titlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 24f.sp
            isFakeBoldText = true
            textAlign = Paint.Align.CENTER
            typeface = Typeface.DEFAULT_BOLD
        }
        private val subtitlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(120, 180, 200, 255)
            textSize = 14f.sp
            textAlign = Paint.Align.CENTER
        }
        private val statusPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(180, 255, 255, 255)
            textSize = 13f.sp
            textAlign = Paint.Align.CENTER
        }
        private val countPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(200, 255, 255, 255)
            textSize = 11f.sp
            textAlign = Paint.Align.CENTER
        }

        // Buttons
        private val btnGradient = GradientDrawable(
            GradientDrawable.Orientation.LEFT_RIGHT,
            intArrayOf(Color.rgb(70, 130, 255), Color.rgb(130, 90, 255))
        )
        private val btnDisabledGradient = GradientDrawable(
            GradientDrawable.Orientation.LEFT_RIGHT,
            intArrayOf(
                Color.argb(80, 70, 130, 255),
                Color.argb(80, 130, 90, 255)
            )
        )
        private val btnSuccessGradient = GradientDrawable(
            GradientDrawable.Orientation.LEFT_RIGHT,
            intArrayOf(Color.rgb(50, 200, 130), Color.rgb(80, 180, 100))
        )
        private val btnTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 16f.sp
            isFakeBoldText = true
            textAlign = Paint.Align.CENTER
            typeface = Typeface.DEFAULT_BOLD
        }
        private val btnBackBorderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(40, 255, 255, 255)
            style = Paint.Style.STROKE
            strokeWidth = 1.5f.dp
        }
        private val btnBackTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(180, 200, 210, 230)
            textSize = 14f.sp
            textAlign = Paint.Align.CENTER
        }

        // Particle
        private val particlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
        }

        // Success overlay
        private val successOverlayPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(100, 40, 200, 120)
        }

        // Failure overlay
        private val failureOverlayPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(120, 200, 40, 40)
        }

        init {
            // Entrance fade-in
            fadeInAnimator = ValueAnimator.ofFloat(0f, 1f).apply {
                duration = 600
                interpolator = DecelerateInterpolator(1.5f)
                addUpdateListener {
                    alphaIn = it.animatedValue as Float
                    invalidate()
                }
                start()
            }
        }

        // ═══════════════════════════════════════════════
        // Public control methods
        // ═══════════════════════════════════════════════

        fun startVerifying() {
            isVerifyingLocal = true
            matchCount = 0
            failCount = 0
            scanPhase = 0f
            pulsePhase = 0f
            showSuccess = false
            showFailure = false
            particles.clear()
            startScanAnimation()
            startPulseAnimation()
            invalidate()
        }

        fun stopAnimation() {
            isVerifyingLocal = false
            scanAnimator?.cancel()
            pulseAnimator?.cancel()
            successAnimator?.cancel()
            particleAnimator?.cancel()
            invalidate()
        }

        fun updateMatchCount(matched: Int, failed: Int) {
            matchCount = matched
            failCount = failed
            if (matched > 0) {
                // Brief dot glow pulse
                postInvalidate()
            }
            invalidate()
        }

        fun onVerifySuccess() {
            isVerifyingLocal = false
            scanAnimator?.cancel()
            pulseAnimator?.cancel()
            showSuccess = true
            showFailure = false
            successPhase = 0f
            spawnParticles()
            startSuccessAnimation()
            startParticleAnimation()
            invalidate()
        }

        fun onVerifyFailed() {
            isVerifyingLocal = false
            scanAnimator?.cancel()
            pulseAnimator?.cancel()
            showSuccess = false
            showFailure = true
            invalidate()
        }

        // ═══════════════════════════════════════════════
        // Animations
        // ═══════════════════════════════════════════════

        private fun startScanAnimation() {
            scanAnimator = ValueAnimator.ofFloat(0f, 1f).apply {
                duration = 2000
                repeatCount = ValueAnimator.INFINITE
                repeatMode = ValueAnimator.REVERSE
                interpolator = AccelerateDecelerateInterpolator()
                addUpdateListener {
                    scanPhase = it.animatedValue as Float
                    invalidate()
                }
                start()
            }
        }

        private fun startPulseAnimation() {
            pulseAnimator = ValueAnimator.ofFloat(0f, 1f).apply {
                duration = 1500
                repeatCount = ValueAnimator.INFINITE
                interpolator = LinearInterpolator()
                addUpdateListener {
                    pulsePhase = it.animatedValue as Float
                    glowAlpha = 0.3f + 0.25f * Math.sin(
                        (it.animatedValue as Float * Math.PI * 2).toDouble()
                    ).toFloat()
                    invalidate()
                }
                start()
            }
        }

        private fun startSuccessAnimation() {
            successAnimator = ValueAnimator.ofFloat(0f, 1f).apply {
                duration = 1200
                interpolator = DecelerateInterpolator()
                addUpdateListener {
                    successPhase = it.animatedValue as Float
                    invalidate()
                }
                start()
            }
        }

        private fun startParticleAnimation() {
            particleAnimator = ValueAnimator.ofFloat(0f, 1f).apply {
                duration = 2000
                interpolator = LinearInterpolator()
                addUpdateListener {
                    val progress = it.animatedValue as Float
                    // Update particles
                    for (p in particles) {
                        p.x += p.vx * 0.016f
                        p.y += p.vy * 0.016f
                        p.vx *= 0.98f
                        p.vy += 80f * 0.016f // gravity
                        p.alpha = (1f - progress).coerceIn(0f, 1f)
                        p.scale = 1f - progress * 0.8f
                    }
                    // Remove faded particles
                    particles.removeAll { it.alpha <= 0.01f }
                    invalidate()
                }
                start()
            }
        }

        private fun spawnParticles() {
            val w = width.toFloat()
            val h = height.toFloat()
            val cx = w / 2f
            val cy = h * 0.45f
            val rng = java.util.Random()

            val colors = intArrayOf(
                Color.rgb(80, 220, 160),
                Color.rgb(60, 200, 240),
                Color.rgb(140, 180, 255),
                Color.rgb(100, 255, 200),
                Color.rgb(200, 220, 100),
                Color.rgb(255, 180, 100),
                Color.rgb(255, 140, 200)
            )

            for (i in 0 until 40) {
                val angle = rng.nextFloat() * Math.PI * 2
                val speed = 200f + rng.nextFloat() * 450f
                particles.add(
                    Particle(
                        x = cx + rng.nextFloat() * 60f - 30f,
                        y = cy + rng.nextFloat() * 60f - 30f,
                        vx = Math.cos(angle).toFloat() * speed,
                        vy = Math.sin(angle).toFloat() * speed - 100f,
                        radius = 3f.dp + rng.nextFloat() * 5f.dp,
                        color = colors[rng.nextInt(colors.size)],
                        alpha = 1f,
                        scale = 1f
                    )
                )
            }
        }

        // ═══════════════════════════════════════════════
        // Touch handling
        // ═══════════════════════════════════════════════

        override fun onTouchEvent(event: MotionEvent): Boolean {
            if (event.action == MotionEvent.ACTION_UP) {
                val x = event.x
                val y = event.y
                if (btnPrimaryRect.contains(x, y)) {
                    if (showSuccess || showFailure) {
                        onBackClick?.invoke()
                    } else if (!isVerifyingLocal) {
                        onStartClick?.invoke()
                    }
                    return true
                }
                if (btnBackRect.contains(x, y)) {
                    onBackClick?.invoke()
                    return true
                }
            }
            return true
        }

        // ═══════════════════════════════════════════════
        // Drawing
        // ═══════════════════════════════════════════════

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val w = width.toFloat()
            val h = height.toFloat()

            if (alphaIn < 1f) {
                canvas.drawARGB((40 * (1f - alphaIn)).toInt(), 0, 0, 0)
            }

            // ── 1. Camera frame area (3:4 portrait oval) ──
            val frameWidth = minOf(w * 0.78f, h * 0.5f * 3f / 4f)
            val frameHeight = frameWidth * 4f / 3f
            val frameLeft = (w - frameWidth) / 2f
            val frameTop = h * 0.14f
            val frameRight = frameLeft + frameWidth
            val frameBottom = frameTop + frameHeight

            val frameRect = RectF(frameLeft, frameTop, frameRight, frameBottom)

            // ── 2. Dim area around face frame (camera preview shows through) ──
            val dimPath = Path().apply {
                addRect(0f, 0f, w, h, Path.Direction.CW)
                addOval(frameRect, Path.Direction.CCW)
            }
            canvas.drawPath(dimPath, dimPaint)

            // ── 4. Pulsing glow ring around frame ──
            if (isVerifyingLocal) {
                pulseRingPaint.apply {
                    color = Color.argb((80 + 40 * glowAlpha).toInt(), 100, 170, 255)
                    strokeWidth = ((4f + 2f * glowAlpha) * resources.displayMetrics.density)
                }
                val pulseRect = RectF(
                    frameLeft - 8f.dp,
                    frameTop - 8f.dp,
                    frameRight + 8f.dp,
                    frameBottom + 8f.dp
                )
                canvas.drawOval(pulseRect, pulseRingPaint)

                // Second outer ring (larger, more transparent)
                pulseRingPaint.apply {
                    color = Color.argb((30 + 20 * glowAlpha).toInt(), 100, 170, 255)
                    strokeWidth = 2f.dp
                }
                val outerPulseRect = RectF(
                    frameLeft - 16f.dp,
                    frameTop - 16f.dp,
                    frameRight + 16f.dp,
                    frameBottom + 16f.dp
                )
                canvas.drawOval(outerPulseRect, pulseRingPaint)
            }

            // ── 5. Oval border ──
            val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color = Color.argb(100, 255, 255, 255)
                style = Paint.Style.STROKE
                strokeWidth = 1.5f.dp
            }
            canvas.drawOval(frameRect, borderPaint)

            // ── 6. Glowing corner brackets (inset for oval shape) ──
            val cornerLen = frameWidth * 0.10f
            val cornerGap = 18f.dp
            val cornerAlpha = if (isVerifyingLocal) (200 + 55 * glowAlpha).toInt() else 160

            // Draw glow first (larger stroke)
            cornerGlowPaint.alpha = (cornerAlpha / 3).coerceIn(30, 90)
            cornerPaint.alpha = cornerAlpha.coerceIn(0, 255)

            drawCornerBracket(
                canvas, frameLeft + cornerGap, frameTop + cornerGap,
                cornerLen, true, true
            )
            drawCornerBracket(
                canvas, frameRight - cornerGap, frameTop + cornerGap,
                cornerLen, false, true
            )
            drawCornerBracket(
                canvas, frameLeft + cornerGap, frameBottom - cornerGap,
                cornerLen, true, false
            )
            drawCornerBracket(
                canvas, frameRight - cornerGap, frameBottom - cornerGap,
                cornerLen, false, false
            )

            // ── 7. Scanning line ──
            if (isVerifyingLocal) {
                val scanY =
                    frameTop + frameHeight * 0.05f + frameHeight * 0.9f * scanPhase
                val scanHeight = 4f.dp
                canvas.save()
                canvas.clipPath(Path().apply { addOval(frameRect, Path.Direction.CW) })

                // Gradient scan line
                val scanRect = RectF(frameLeft, scanY - scanHeight, frameRight, scanY + scanHeight)
                val scanGradientLocal = LinearGradient(
                    0f, scanY - scanHeight, 0f, scanY + scanHeight,
                    intArrayOf(
                        Color.TRANSPARENT,
                        Color.argb(60, 120, 200, 255),
                        Color.argb(150, 140, 220, 255),
                        Color.argb(60, 120, 200, 255),
                        Color.TRANSPARENT
                    ),
                    floatArrayOf(0f, 0.25f, 0.5f, 0.75f, 1f),
                    Shader.TileMode.CLAMP
                )
                val scanLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    shader = scanGradientLocal
                    style = Paint.Style.FILL
                }
                canvas.drawRect(scanRect, scanLinePaint)
                canvas.restore()
            }

            // ── 8. Title & Status ──
            canvas.drawText("人脸验证", w / 2f, h * 0.06f, titlePaint)

            val statusText: String
            val statusColor: Int
            when {
                showSuccess -> {
                    statusText = "✓ 验证通过"
                    statusColor = Color.rgb(80, 220, 150)
                }
                showFailure -> {
                    statusText = "✗ 验证失败"
                    statusColor = Color.rgb(255, 100, 100)
                }
                isVerifyingLocal -> {
                    statusText = "正在进行人脸验证..."
                    statusColor = Color.argb(180, 140, 200, 255)
                }
                else -> {
                    statusText = "请将脸部对准摄像头进行验证"
                    statusColor = Color.argb(150, 180, 200, 230)
                }
            }
            statusPaint.color = statusColor
            canvas.drawText(statusText, w / 2f, h * 0.06f + 28f.dp, statusPaint)

            // ── 9. Progress Dots ──
            val dotCount = 5
            val dotRadius = 7f.dp
            val dotSpacing = 24f.dp
            val dotsTotalWidth = dotCount * (dotRadius * 2) + (dotCount - 1) * dotSpacing
            val dotsStartX = (w - dotsTotalWidth) / 2f + dotRadius
            val dotsY = frameBottom + 22f.dp

            for (i in 0 until dotCount) {
                val cx = dotsStartX + i * (dotRadius * 2 + dotSpacing)
                val isActive = i < matchCount
                val isFailed = failCount >= MAX_FAILS

                if (isFailed) {
                    // Failed dots pulse red
                    val pulseScale = if (i % 2 == 0) 1f + 0.3f * glowAlpha else 1f
                    canvas.drawCircle(cx, dotsY, dotRadius * pulseScale, dotFailedPaint)
                } else if (isActive) {
                    // Active dot with glow
                    val pulseScale = 1f + 0.25f * glowAlpha
                    canvas.drawCircle(cx, dotsY, dotRadius * 1.6f * pulseScale, dotActiveGlowPaint)
                    canvas.drawCircle(cx, dotsY, dotRadius * pulseScale, dotActivePaint)
                } else {
                    canvas.drawCircle(cx, dotsY, dotRadius, dotInactivePaint)
                }
            }

            // Progress count text
            val countStr = if (matchCount >= MATCH_MIN) "验证完成" else "$matchCount / $MATCH_MIN 次匹配"
            canvas.drawText(countStr, w / 2f, dotsY + dotRadius + 18f.dp, countPaint)

            // ── 10. Success overlay ──
            if (showSuccess && successPhase > 0f) {
                // Green flash
                successOverlayPaint.alpha = (100 * (1f - successPhase)).toInt().coerceIn(0, 100)
                canvas.drawPaint(successOverlayPaint)

                // Checkmark circle
                val checkCX = w / 2f
                val checkCY = h * 0.38f
                val checkRadius = 36f.dp * successPhase

                // Outer ring
                val checkRingPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    color = Color.argb(
                        (200 * (1f - successPhase * 0.5f)).toInt().coerceIn(30, 200),
                        80, 220, 150
                    )
                    style = Paint.Style.STROKE
                    strokeWidth = 5f.dp
                }
                canvas.drawCircle(checkCX, checkCY, checkRadius, checkRingPaint)

                // Filled circle
                val checkFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    color = Color.argb(
                        (180 * (1f - successPhase * 0.3f)).toInt().coerceIn(40, 180),
                        80, 220, 150
                    )
                    style = Paint.Style.FILL
                }
                canvas.drawCircle(checkCX, checkCY, checkRadius * 0.85f, checkFillPaint)

                // Checkmark
                val checkPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    color = Color.WHITE
                    style = Paint.Style.STROKE
                    strokeWidth = 5f.dp
                    strokeCap = Paint.Cap.ROUND
                    strokeJoin = Paint.Join.ROUND
                }
                val checkPath = Path().apply {
                    moveTo(checkCX - 14f.dp, checkCY)
                    lineTo(checkCX - 4f.dp, checkCY + 10f.dp)
                    lineTo(checkCX + 14f.dp, checkCY - 8f.dp)
                }
                canvas.drawPath(checkPath, checkPaint)

                // "验证通过" text
                val successTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    color = Color.WHITE
                    textSize = 20f.sp
                    isFakeBoldText = true
                    textAlign = Paint.Align.CENTER
                    alpha = (255 * successPhase).toInt().coerceIn(0, 255)
                }
                canvas.drawText("验证通过", w / 2f, checkCY + checkRadius + 30f.dp, successTextPaint)
            }

            // ── 11. Failure overlay ──
            if (showFailure) {
                // Red flash
                failureOverlayPaint.alpha = 80
                canvas.drawPaint(failureOverlayPaint)

                // X circle
                val failCX = w / 2f
                val failCY = h * 0.38f
                val failRadius = 36f.dp

                val failFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    color = Color.argb(150, 220, 50, 50)
                    style = Paint.Style.FILL
                }
                canvas.drawCircle(failCX, failCY, failRadius, failFillPaint)

                // X
                val xPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    color = Color.WHITE
                    style = Paint.Style.STROKE
                    strokeWidth = 5f.dp
                    strokeCap = Paint.Cap.ROUND
                }
                canvas.drawLine(
                    failCX - 10f.dp, failCY - 10f.dp,
                    failCX + 10f.dp, failCY + 10f.dp, xPaint
                )
                canvas.drawLine(
                    failCX + 10f.dp, failCY - 10f.dp,
                    failCX - 10f.dp, failCY + 10f.dp, xPaint
                )

                val failTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    color = Color.argb(220, 255, 160, 160)
                    textSize = 20f.sp
                    isFakeBoldText = true
                    textAlign = Paint.Align.CENTER
                }
                canvas.drawText("验证失败", w / 2f, failCY + failRadius + 30f.dp, failTextPaint)
            }

            // ── 12. Particles (success) ──
            for (p in particles) {
                particlePaint.apply {
                    color = p.color
                    alpha = (255 * p.alpha).toInt().coerceIn(0, 255)
                }
                canvas.drawCircle(p.x, p.y, p.radius * p.scale, particlePaint)
            }

            // ── 13. Buttons ──
            val btnWidth = w * 0.78f
            val btnHeight = 46f.dp
            val btnStartX = (w - btnWidth) / 2f
            val primaryBtnY = dotsY + dotRadius + 42f.dp
            val backBtnY = primaryBtnY + btnHeight + 14f.dp

            btnPrimaryRect.set(btnStartX, primaryBtnY, btnStartX + btnWidth, primaryBtnY + btnHeight)

            // Primary button
            val primaryGradient = when {
                showSuccess -> btnSuccessGradient
                isVerifyingLocal -> btnDisabledGradient
                else -> btnGradient
            }
            primaryGradient.setBounds(
                btnStartX.toInt(),
                primaryBtnY.toInt(),
                (btnStartX + btnWidth).toInt(),
                (primaryBtnY + btnHeight).toInt()
            )
            primaryGradient.cornerRadius = 14f.dp
            primaryGradient.draw(canvas)

            val btnText = when {
                showSuccess || showFailure -> "返回"
                isVerifyingLocal -> "正在验证..."
                else -> "开始人脸验证"
            }
            if (isVerifyingLocal) btnTextPaint.alpha = 100 else btnTextPaint.alpha = 255
            canvas.drawText(
                btnText,
                w / 2f,
                primaryBtnY + btnHeight / 2f - (btnTextPaint.descent() + btnTextPaint.ascent()) / 2f,
                btnTextPaint
            )

            // Back button (hidden when success/failure)
            if (!showSuccess && !showFailure) {
                btnBackRect.set(btnStartX, backBtnY, btnStartX + btnWidth, backBtnY + btnHeight)
                val backGradient = GradientDrawable().apply {
                    setColor(Color.TRANSPARENT)
                    cornerRadius = 12f.dp
                    setStroke(1.5f.dp.toInt(), Color.argb(40, 200, 210, 230))
                }
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
            }
        }

        private fun drawCornerBracket(
            canvas: Canvas,
            x: Float, y: Float,
            len: Float,
            isLeft: Boolean, isTop: Boolean
        ) {
            val dirX = if (isLeft) 1f else -1f
            val dirY = if (isTop) 1f else -1f

            // Glow
            canvas.drawLine(x, y, x + len * dirX, y, cornerGlowPaint)
            canvas.drawLine(x, y, x, y + len * dirY, cornerGlowPaint)

            // Main
            canvas.drawLine(x, y, x + len * dirX, y, cornerPaint)
            canvas.drawLine(x, y, x, y + len * dirY, cornerPaint)
        }

        override fun onDetachedFromWindow() {
            super.onDetachedFromWindow()
            scanAnimator?.cancel()
            pulseAnimator?.cancel()
            successAnimator?.cancel()
            particleAnimator?.cancel()
            fadeInAnimator?.cancel()
        }
    }

    // ═══════════════════════════════════════════════════
    // Particle data class
    // ═══════════════════════════════════════════════════

    data class Particle(
        var x: Float,
        var y: Float,
        var vx: Float,
        var vy: Float,
        var radius: Float,
        var color: Int,
        var alpha: Float,
        var scale: Float
    )

    // ─── Extension properties for dp/sp conversion ───

    private val Float.dp: Float get() = this * resources.displayMetrics.density
    private val Float.sp: Float get() = this * resources.displayMetrics.scaledDensity
}
