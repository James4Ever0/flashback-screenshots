package com.openmule.screenshotworker.service

import android.app.Notification
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.widget.Toast
import androidx.core.app.NotificationCompat
import com.openmule.screenshotworker.MainActivity
import com.openmule.screenshotworker.ScreenshotWorkerApp
import com.openmule.screenshotworker.data.ConfigRepository
import com.openmule.screenshotworker.data.DiagnosticRepository
import com.openmule.screenshotworker.data.LogRepository
import com.openmule.screenshotworker.data.ServiceStatePersistence
import com.openmule.screenshotworker.storage.DiskBuffer
import com.openmule.screenshotworker.transport.WebSocketTransport
import kotlinx.coroutines.*
import java.io.ByteArrayOutputStream
import java.io.PrintWriter
import java.io.StringWriter
import java.util.concurrent.atomic.AtomicInteger

/**
 * Foreground service that captures screenshots via MediaProjection
 * and sends them to the server via WebSocket.
 *
 * Persistence strategy:
 * 1. startForeground() with ongoing notification (required by Android 8+)
 * 2. START_REDELIVER_INTENT — OS redelivers last intent on restart
 * 3. WakeLock (PARTIAL) — keeps CPU alive during capture
 * 4. Broadcast restart on destroy — triggers BootReceiver to attempt recovery
 * 5. AlarmManager keep-alive — periodic check every 60s via setExactAndAllowWhileIdle
 */
class ScreenshotService : Service() {

    private val logger = LogRepository.getInstance()
    private val diag = DiagnosticRepository.getInstance()
    private val serviceScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    private var mediaProjection: MediaProjection? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null

    private lateinit var config: ConfigRepository
    private lateinit var diskBuffer: DiskBuffer
    private var transport: WebSocketTransport? = null
    private var captureJob: Job? = null
    private var wakeLock: PowerManager.WakeLock? = null
    private var heartbeatJob: Job? = null
    private var screenStateHelper: ScreenStateHelper? = null

    /** When this service instance started. */
    private var serviceStartTime: Long = 0

    /** Total captures this session — thread-safe since ImageReader callback runs on main looper
     * but diagnostic reads may happen on other threads. */
    private val totalCaptures = AtomicInteger(0)

    /** Watchdog: logs warning if no captures within first N seconds. */
    private var watchdogJob: Job? = null

    override fun onCreate() {
        super.onCreate()
        try {
            config = ConfigRepository(this)
            diskBuffer = DiskBuffer(this, config.bufferMaxFiles)
            logger.i(TAG, "Service created")
        } catch (e: Exception) {
            logException("onCreate", e)
            toast("Service create failed: ${e.message}")
            throw e
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        logger.i(TAG, "onStartCommand startId=$startId intent=${intent != null} action=${intent?.action} extras=${intent?.extras?.keySet()?.joinToString()}")

        try {
            // Must start as foreground immediately — required within 5s on Android 8+
            startForeground(NOTIFICATION_ID, createNotification())
        } catch (e: Exception) {
            logException("startForeground", e)
            toast("startForeground failed: ${e.message}")
            stopSelf()
            return START_NOT_STICKY
        }

        try {
            // Handle explicit stop action from notification
            if (intent?.action == ACTION_STOP) {
                logger.i(TAG, "Stop action received")
                stopSelf()
                return START_NOT_STICKY
            }

            // If already running, just re-establish foreground and schedule next alarm
            if (mediaProjection != null) {
                logger.i(TAG, "Service already running, refreshing foreground")
                ServiceKeepAliveHelper.schedule(this)
                return START_REDELIVER_INTENT
            }

            // Read extras defensively
            val resultCode = intent?.getIntExtra(EXTRA_RESULT_CODE, -999) ?: -999
            val data: Intent? = try {
                @Suppress("DEPRECATION")
                intent?.getParcelableExtra(EXTRA_DATA) as? Intent
            } catch (e: Exception) {
                logException("getParcelableExtra", e)
                null
            }

            logger.i(TAG, "Extras read: resultCode=$resultCode data=${data != null}")

            if (resultCode == -999 || data == null) {
                logger.w(TAG, "Missing MediaProjection data — resultCode=$resultCode data=${data != null}. " +
                        "This happens when the OS restarts the service without the original intent.")
                showRecoveryNotification()
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
                return START_NOT_STICKY
            }

            val projectionManager = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
            mediaProjection = projectionManager.getMediaProjection(resultCode, data)

            if (mediaProjection == null) {
                val msg = "MediaProjection is null — permission token may have expired (resultCode=$resultCode)"
                logger.e(TAG, msg)
                diag.update { copy(lastError = msg, lastErrorTime = System.currentTimeMillis()) }
                toast(msg)
                stopSelf()
                return START_NOT_STICKY
            }

            logger.i(TAG, "MediaProjection acquired successfully")

            // Android 14+ (API 34) REQUIRES a callback before createVirtualDisplay().
            // The callback API exists since API 21 so this is safe for all versions.
            mediaProjection?.registerCallback(object : MediaProjection.Callback() {
                override fun onStop() {
                    logger.w(TAG, "MediaProjection stopped by system/user")
                    diag.update {
                        copy(
                            lastError = "MediaProjection stopped by system",
                            lastErrorTime = System.currentTimeMillis()
                        )
                    }
                    // Clean shutdown — don't broadcast restart here, let onDestroy handle it
                    captureJob?.cancel()
                    try {
                        virtualDisplay?.release()
                        imageReader?.close()
                    } catch (_: Exception) {}
                }
            }, Handler(Looper.getMainLooper()))

            serviceStartTime = System.currentTimeMillis()
            totalCaptures.set(0)
            ServiceStatePersistence.markServiceStarted(this, serviceStartTime)
            diag.update {
                copy(
                    serviceAlive = true,
                    serviceStartTime = serviceStartTime,
                    totalCaptures = 0,
                    lastCaptureTime = 0,
                    lastCaptureResolution = "",
                    lastCaptureSizeBytes = 0,
                    websocketConnected = false,
                    lastWebsocketSendTime = 0,
                    totalFramesSent = 0,
                    bufferFileCount = 0,
                    bufferTotalBytes = 0,
                    lastError = "",
                    lastErrorTime = 0
                )
            }

            setupCapture()
            startTransport()
            acquireWakeLock()
            ServiceKeepAliveHelper.schedule(this)
            startDiagnosticUpdates()
            startWatchdog()
            startHeartbeat()

            screenStateHelper = ScreenStateHelper(this).apply { register() }

            logger.i(TAG, "Service fully started")
            return START_REDELIVER_INTENT

        } catch (e: Exception) {
            logException("onStartCommand body", e)
            toast("Service start failed: ${e.message}")
            diag.update { copy(lastError = "onStartCommand: ${e.message}", lastErrorTime = System.currentTimeMillis()) }
            stopSelf()
            return START_NOT_STICKY
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        logger.i(TAG, "Service destroying")
        try {
            val restartIntent = Intent(ACTION_RESTART_SERVICE).apply {
                setPackage(packageName)
            }
            sendBroadcast(restartIntent)
        } catch (e: Exception) {
            logException("sendBroadcast in onDestroy", e)
        }

        try {
            watchdogJob?.cancel()
            captureJob?.cancel()
            heartbeatJob?.cancel()
            serviceScope.cancel()
            transport?.stop()
            virtualDisplay?.release()
            imageReader?.close()
            try {
                mediaProjection?.unregisterCallback(object : MediaProjection.Callback() {})
            } catch (_: Exception) {}
            mediaProjection?.stop()
            releaseWakeLock()
            ServiceKeepAliveHelper.cancel(this)
            screenStateHelper?.unregister()
        } catch (e: Exception) {
            logException("onDestroy cleanup", e)
        }

        val uptime = if (serviceStartTime > 0) System.currentTimeMillis() - serviceStartTime else 0
        logger.i(TAG, "Service uptime: ${uptime}ms, total captures: ${totalCaptures.get()}")
        ServiceStatePersistence.markServiceStopped(this, uptime)
        diag.update {
            copy(
                serviceAlive = false,
                lastSessionUptimeMs = uptime
            )
        }

        super.onDestroy()
    }

    private fun acquireWakeLock() {
        try {
            val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
            wakeLock = pm.newWakeLock(
                PowerManager.PARTIAL_WAKE_LOCK,
                "ScreenshotWorker::ServiceWakeLock"
            ).apply {
                setReferenceCounted(false)
                acquire()
            }
            logger.i(TAG, "Wake lock acquired")
        } catch (e: Exception) {
            logException("acquireWakeLock", e)
        }
    }

    private fun releaseWakeLock() {
        try {
            wakeLock?.let {
                if (it.isHeld) it.release()
                wakeLock = null
                logger.i(TAG, "Wake lock released")
            }
        } catch (e: Exception) {
            logException("releaseWakeLock", e)
        }
    }

    private var lastCaptureTime = 0L

    private fun setupCapture() {
        try {
            val displayMetrics = resources.displayMetrics
            val width = displayMetrics.widthPixels
            val height = displayMetrics.heightPixels
            val density = displayMetrics.densityDpi

            logger.i(TAG, "Display: ${width}x${height} @ ${density}dpi")

            imageReader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2)
            logger.i(TAG, "ImageReader created: ${width}x${height}, maxImages=2")

            imageReader?.setOnImageAvailableListener({ reader ->
                try {
                    // Skip capture when screen is off or device is locked
                    if (screenStateHelper?.canCapture() == false) {
                        reader.acquireLatestImage()?.close()
                        return@setOnImageAvailableListener
                    }

                    val now = System.currentTimeMillis()
                    val minInterval = config.captureIntervalSec * 1000L
                    if (now - lastCaptureTime < minInterval) {
                        reader.acquireLatestImage()?.close()
                        return@setOnImageAvailableListener
                    }
                    lastCaptureTime = now

                    if (wakeLock?.isHeld == false) {
                        acquireWakeLock()
                    }

                    val image = reader.acquireLatestImage()
                    if (image == null) {
                        logger.w(TAG, "acquireLatestImage returned null — no frame available")
                        return@setOnImageAvailableListener
                    }
                    try {
                        logger.d(TAG, "Image acquired: ${image.width}x${image.height}, format=${image.format}")
                        val jpeg = imageToJpeg(image, config.jpegQuality)
                        val timestamp = System.nanoTime()
                        diskBuffer.write(timestamp, jpeg)
                        val count = totalCaptures.incrementAndGet()

                        val resolution = "${width}x${height}"
                        diag.update {
                            copy(
                                lastCaptureTime = now,
                                totalCaptures = count,
                                lastCaptureResolution = resolution,
                                lastCaptureSizeBytes = jpeg.size
                            )
                        }
                        logger.i(TAG, "Captured $resolution frame #$count (${jpeg.size} bytes)")
                    } catch (e: Exception) {
                        logException("capture frame", e)
                        diag.update {
                            copy(
                                lastError = "Capture: ${e.message}",
                                lastErrorTime = System.currentTimeMillis()
                            )
                        }
                    } finally {
                        image.close()
                    }
                } catch (e: Exception) {
                    logException("onImageAvailable callback", e)
                }
            }, Handler(Looper.getMainLooper()))

            virtualDisplay = mediaProjection?.createVirtualDisplay(
                "ScreenshotWorker",
                width, height, density,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                imageReader?.surface, null, null
            )

            if (virtualDisplay != null) {
                logger.i(TAG, "VirtualDisplay created successfully, surface=${imageReader?.surface != null}")
            } else {
                logger.e(TAG, "VirtualDisplay is null — capture will not work!")
            }

            logger.i(TAG, "Capture started, interval: ${config.captureIntervalSec}s")
        } catch (e: Exception) {
            logException("setupCapture", e)
            toast("Capture setup failed: ${e.message}")
            throw e
        }
    }

    private fun startTransport() {
        try {
            transport = WebSocketTransport(
                this,
                config.serverUrl,
                config.clientName,
                diskBuffer,
                config.wifiOnlyUpload
            )
            transport?.start()
        } catch (e: Exception) {
            logException("startTransport", e)
        }
    }

    /** Warns if no captures happen within the first 30 seconds — helps diagnose setup issues. */
    private fun startWatchdog() {
        watchdogJob = serviceScope.launch {
            delay(30_000)
            val count = totalCaptures.get()
            if (count == 0) {
                logger.w(TAG, "WATCHDOG: No captures after 30s. " +
                        "virtualDisplay=${virtualDisplay != null}, " +
                        "imageReader=${imageReader != null}, " +
                        "mediaProjection=${mediaProjection != null}")
                diag.update {
                    copy(
                        lastError = "Watchdog: no captures after 30s",
                        lastErrorTime = System.currentTimeMillis()
                    )
                }
            } else {
                logger.i(TAG, "WATCHDOG: $count captures in first 30s — looks healthy")
            }
        }
    }

    /** Writes a heartbeat every 5s so the UI can detect if we were OS-killed. */
    private fun startHeartbeat() {
        heartbeatJob = serviceScope.launch {
            while (isActive) {
                try {
                    ServiceStatePersistence.updateHeartbeat(this@ScreenshotService)
                    delay(5_000)
                } catch (e: Exception) {
                    logException("heartbeat", e)
                }
            }
        }
    }

    /** Periodically updates buffer stats and websocket status in the diagnostic state. */
    private fun startDiagnosticUpdates() {
        serviceScope.launch {
            while (isActive) {
                try {
                    delay(2000)
                    val fileCount = diskBuffer.count()
                    val totalBytes = diskBuffer.totalSizeBytes()
                    diag.update {
                        copy(
                            bufferFileCount = fileCount,
                            bufferTotalBytes = totalBytes,
                            bufferTotalWritten = diskBuffer.totalWritten
                        )
                    }
                } catch (e: Exception) {
                    logException("diagnostic update", e)
                }
            }
        }
    }

    private fun imageToJpeg(image: Image, quality: Int): ByteArray {
        val width = image.width
        val height = image.height
        val planes = image.planes
        val buffer = planes[0].buffer
        val pixelStride = planes[0].pixelStride
        val rowStride = planes[0].rowStride
        val rowPadding = rowStride - pixelStride * width

        buffer.rewind()

        val bitmap = Bitmap.createBitmap(
            width + rowPadding / pixelStride,
            height,
            Bitmap.Config.ARGB_8888
        )
        bitmap.copyPixelsFromBuffer(buffer)

        val cropped = if (bitmap.width != width) {
            Bitmap.createBitmap(bitmap, 0, 0, width, height).also { bitmap.recycle() }
        } else bitmap

        val output = ByteArrayOutputStream()
        cropped.compress(Bitmap.CompressFormat.JPEG, quality, output)
        cropped.recycle()
        return output.toByteArray()
    }

    private fun createNotification(): Notification {
        val channelId = ScreenshotWorkerApp.CHANNEL_SERVICE

        val openAppIntent = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE
        )

        val stopIntent = Intent(this, ScreenshotService::class.java).apply {
            action = ACTION_STOP
        }
        val stopPendingIntent = PendingIntent.getService(
            this, 1, stopIntent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        return NotificationCompat.Builder(this, channelId)
            .setContentTitle("Screenshot Worker")
            .setContentText("Running • ${config.clientName}")
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setContentIntent(openAppIntent)
            .setOngoing(true)
            .setShowWhen(false)
            .addAction(android.R.drawable.ic_media_pause, "Stop", stopPendingIntent)
            .build()
    }

    private fun showRecoveryNotification() {
        try {
            val channelId = ScreenshotWorkerApp.CHANNEL_RECOVERY
            val pendingIntent = PendingIntent.getActivity(
                this, 1,
                Intent(this, MainActivity::class.java).apply {
                    flags = Intent.FLAG_ACTIVITY_NEW_TASK or
                            Intent.FLAG_ACTIVITY_CLEAR_TOP or
                            Intent.FLAG_ACTIVITY_REORDER_TO_FRONT
                    putExtra(EXTRA_AUTO_START, true)
                },
                PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
            )

            val notification = NotificationCompat.Builder(this, channelId)
                .setContentTitle("Screenshot Worker Stopped")
                .setContentText("Tap to reopen and restart capture")
                .setSmallIcon(android.R.drawable.ic_dialog_alert)
                .setContentIntent(pendingIntent)
                .setAutoCancel(true)
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .build()

            val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            nm.notify(RECOVERY_NOTIFICATION_ID, notification)
        } catch (e: Exception) {
            logException("showRecoveryNotification", e)
        }
    }

    /** Logs the full exception with stack trace — never swallowed. */
    private fun logException(context: String, e: Throwable) {
        val sw = StringWriter()
        e.printStackTrace(PrintWriter(sw))
        logger.e(TAG, "EXCEPTION in $context: ${e.message}\n$sw")
    }

    /** Shows a toast on the main thread. */
    private fun toast(message: String) {
        try {
            Handler(Looper.getMainLooper()).post {
                Toast.makeText(this, message, Toast.LENGTH_LONG).show()
            }
        } catch (e: Exception) {
            logger.e(TAG, "Toast failed: ${e.message}")
        }
    }

    companion object {
        private const val TAG = "ScreenshotService"
        private const val NOTIFICATION_ID = 1
        private const val RECOVERY_NOTIFICATION_ID = 2
        const val EXTRA_RESULT_CODE = "result_code"
        const val EXTRA_DATA = "data"
        const val ACTION_RESTART_SERVICE = "com.openmule.screenshotworker.RESTART_SERVICE"
        const val ACTION_STOP = "com.openmule.screenshotworker.STOP_SERVICE"
        const val EXTRA_AUTO_START = "auto_start"

        fun start(context: Context, resultCode: Int, data: Intent) {
            try {
                val intent = Intent(context, ScreenshotService::class.java).apply {
                    putExtra(EXTRA_RESULT_CODE, resultCode)
                    putExtra(EXTRA_DATA, data)
                }
                LogRepository.getInstance().i(TAG, "startForegroundService called with resultCode=$resultCode")
                context.startForegroundService(intent)
            } catch (e: Exception) {
                val sw = StringWriter()
                e.printStackTrace(PrintWriter(sw))
                LogRepository.getInstance().e(TAG, "startForegroundService FAILED: ${e.message}\n$sw")
                try {
                    // Fallback: try regular startService
                    val intent = Intent(context, ScreenshotService::class.java).apply {
                        putExtra(EXTRA_RESULT_CODE, resultCode)
                        putExtra(EXTRA_DATA, data)
                    }
                    LogRepository.getInstance().w(TAG, "Falling back to startService")
                    context.startService(intent)
                } catch (e2: Exception) {
                    val sw2 = StringWriter()
                    e2.printStackTrace(PrintWriter(sw2))
                    LogRepository.getInstance().e(TAG, "startService also FAILED: ${e2.message}\n$sw2")
                }
            }
        }

        fun stop(context: Context) {
            try {
                context.stopService(Intent(context, ScreenshotService::class.java))
            } catch (e: Exception) {
                val sw = StringWriter()
                e.printStackTrace(PrintWriter(sw))
                LogRepository.getInstance().e(TAG, "stopService FAILED: ${e.message}\n$sw")
            }
        }
    }
}
