package com.openmule.screenshotworker.ui.viewmodel

import android.app.Activity
import android.app.Application
import android.content.Context
import android.content.Intent
import android.media.projection.MediaProjectionManager
import android.widget.Toast
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.openmule.screenshotworker.data.ConfigRepository
import com.openmule.screenshotworker.data.DiagnosticRepository
import com.openmule.screenshotworker.data.LogRepository
import com.openmule.screenshotworker.data.ServiceStatePersistence
import com.openmule.screenshotworker.service.BatteryOptimizationHelper
import com.openmule.screenshotworker.service.ScreenshotService
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import java.io.PrintWriter
import java.io.StringWriter

/**
 * ViewModel for the main screen. Bridges UI, Config, Service state, and Diagnostics.
 */
class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val config = ConfigRepository(application)
    private val logger = LogRepository.getInstance()
    private val diagRepo = DiagnosticRepository.getInstance()

    val logs = logger.logs.stateIn(
        viewModelScope,
        SharingStarted.WhileSubscribed(5000),
        emptyList()
    )

    /** Live diagnostic state from the service */
    val diagnostics = diagRepo.state.stateIn(
        viewModelScope,
        SharingStarted.WhileSubscribed(5000),
        DiagnosticRepository.DiagnosticState()
    )

    var isRunning by mutableStateOf(false)
        private set

    /** Set by MainActivity when recovery notification is tapped — triggers auto-start in UI. */
    var pendingAutoStart by mutableStateOf(false)

    var serverUrl by mutableStateOf(config.serverUrl)
    var captureInterval by mutableStateOf(config.captureIntervalSec.toString())
    var jpegQuality by mutableStateOf(config.jpegQuality.toString())
    var bufferMaxFiles by mutableStateOf(config.bufferMaxFiles.toString())
    var clientName by mutableStateOf(config.clientName)
    var autoStart by mutableStateOf(config.autoStart)
    var diagnosticRefreshInterval by mutableStateOf(config.diagnosticRefreshIntervalSec.toString())
    var wifiOnlyUpload by mutableStateOf(config.wifiOnlyUpload)

    var showPermissionDenied by mutableStateOf(false)
    var batteryOptimized by mutableStateOf(false)
        private set

    /** Android 13+ requires runtime notification permission for foreground services */
    var notificationPermissionGranted by mutableStateOf(true)
        private set

    init {
        logger.i(TAG, "ViewModel created")
        checkBatteryStatus(getApplication())
        // Guard against stale state after Android Studio deploys / process kills
        val staleCleared = ServiceStatePersistence.clearIfStale(getApplication())
        if (staleCleared) {
            logger.w(TAG, "Cleared stale service state on ViewModel init")
        }
        syncServiceStateFromDisk()
        // Periodic refresh: read heartbeat from disk to detect OS kills
        viewModelScope.launch {
            while (true) {
                delay(5_000)
                syncServiceStateFromDisk()
            }
        }
        // Also react to in-memory state changes
        viewModelScope.launch {
            diagnostics.collect { diag ->
                if (isRunning && !diag.serviceAlive && diag.lastSessionUptimeMs > 0) {
                    logger.w(TAG, "Service died unexpectedly — resetting UI state")
                    isRunning = false
                }
            }
        }
    }

    /** Read the persisted service state (heartbeat + start time) and update UI.
     *  This survives process death — the file is written by ScreenshotService. */
    private fun syncServiceStateFromDisk() {
        try {
            val ctx = getApplication<Application>()
            val alive = ServiceStatePersistence.isServiceAlive(ctx)
            val startTime = ServiceStatePersistence.getStartTime(ctx)
            val lastUptime = ServiceStatePersistence.getLastSessionUptime(ctx)

            // Sync isRunning with reality
            isRunning = alive

            // Update diagnostic repository with persisted values
            diagRepo.update {
                copy(
                    serviceAlive = alive,
                    serviceStartTime = if (alive) startTime else serviceStartTime,
                    lastSessionUptimeMs = if (!alive) lastUptime else lastSessionUptimeMs
                )
            }
        } catch (e: Exception) {
            logException("syncServiceStateFromDisk", e)
        }
    }

    fun onNotificationPermissionChecked(granted: Boolean) {
        notificationPermissionGranted = granted
        if (!granted) {
            logger.w(TAG, "POST_NOTIFICATIONS not granted — foreground service cannot start")
        } else {
            logger.i(TAG, "POST_NOTIFICATIONS granted")
        }
    }

    fun checkBatteryStatus(context: Context) {
        try {
            batteryOptimized = !BatteryOptimizationHelper.isIgnoringBatteryOptimizations(context)
            if (batteryOptimized) {
                logger.w(TAG, "Battery optimization is enabled — service may be killed by OS")
            } else {
                logger.i(TAG, "Battery optimization disabled")
            }
        } catch (e: Exception) {
            logException("checkBatteryStatus", e)
        }
    }

    fun requestBatteryOptimization(activity: Activity): Boolean {
        return try {
            val launched = BatteryOptimizationHelper.requestIgnoreBatteryOptimizations(activity)
            if (launched) {
                logger.i(TAG, "Opened battery optimization settings")
            }
            launched
        } catch (e: Exception) {
            logException("requestBatteryOptimization", e)
            false
        }
    }

    fun saveConfig() {
        try {
            config.serverUrl = serverUrl
            config.captureIntervalSec = captureInterval.toIntOrNull() ?: 5
            config.jpegQuality = jpegQuality.toIntOrNull() ?: 80
            config.bufferMaxFiles = bufferMaxFiles.toIntOrNull() ?: 100
            config.clientName = clientName
            config.autoStart = autoStart
            config.diagnosticRefreshIntervalSec = diagnosticRefreshInterval.toIntOrNull() ?: 3
            config.wifiOnlyUpload = wifiOnlyUpload
            logger.i(TAG, "Config saved")
        } catch (e: Exception) {
            logException("saveConfig", e)
        }
    }

    fun requestMediaProjection(activity: Activity): Intent? {
        return try {
            // Critical: cannot start foreground service without notification permission on API 33+
            if (!notificationPermissionGranted) {
                logger.e(TAG, "Cannot start: notification permission not granted")
                toast(activity, "Notification permission required")
                return null
            }
            val mgr = activity.getSystemService(android.content.Context.MEDIA_PROJECTION_SERVICE)
                    as MediaProjectionManager
            mgr.createScreenCaptureIntent()
        } catch (e: Exception) {
            logException("requestMediaProjection", e)
            toast(activity, "Failed to request capture: ${e.message}")
            null
        }
    }

    fun onMediaProjectionResult(resultCode: Int, data: Intent?) {
        val ctx = getApplication<Application>()
        logger.i(TAG, "onMediaProjectionResult resultCode=$resultCode data=${data != null}")

        if (resultCode == Activity.RESULT_OK && data != null) {
            try {
                logger.i(TAG, "MediaProjection granted, starting service")
                isRunning = true
                showPermissionDenied = false
                ScreenshotService.start(ctx, resultCode, data)
            } catch (e: Exception) {
                logException("onMediaProjectionResult start", e)
                toast(ctx, "Failed to start service: ${e.message}")
                isRunning = false
            }
        } else {
            logger.e(TAG, "MediaProjection denied — resultCode=$resultCode")
            isRunning = false
            showPermissionDenied = true
        }
    }

    fun stopService() {
        val ctx = getApplication<Application>()
        try {
            logger.i(TAG, "Stopping service")
            isRunning = false
            ScreenshotService.stop(ctx)
        } catch (e: Exception) {
            logException("stopService", e)
            toast(ctx, "Failed to stop service: ${e.message}")
        }
    }

    fun clearLogs() {
        try {
            logger.clear()
        } catch (e: Exception) {
            logException("clearLogs", e)
        }
    }

    /** Format milliseconds into a single natural-language unit:
     *  "45 seconds", "3 minutes", "2 hours", "5 days"
     */
    fun formatUptime(ms: Long): String {
        if (ms <= 0) return "—"
        val seconds = ms / 1000
        val mins = seconds / 60
        val hrs = mins / 60
        val days = hrs / 24

        return when {
            days >= 1 -> "$days day${if (days > 1) "s" else ""}"
            hrs >= 1 -> "$hrs hour${if (hrs > 1) "s" else ""}"
            mins >= 1 -> "$mins minute${if (mins > 1) "s" else ""}"
            else -> "$seconds second${if (seconds != 1L) "s" else ""}"
        }
    }

    /** Format bytes into human-readable string */
    fun formatBytes(bytes: Long): String {
        if (bytes < 1024) return "$bytes B"
        if (bytes < 1024 * 1024) return "%.1f KB".format(bytes / 1024.0)
        return "%.2f MB".format(bytes / (1024.0 * 1024.0))
    }

    private fun logException(context: String, e: Throwable) {
        val sw = StringWriter()
        e.printStackTrace(PrintWriter(sw))
        logger.e(TAG, "EXCEPTION in $context: ${e.message}\n$sw")
    }

    private fun toast(context: Context, message: String) {
        try {
            Toast.makeText(context, message, Toast.LENGTH_LONG).show()
        } catch (e: Exception) {
            logger.e(TAG, "Toast failed: ${e.message}")
        }
    }

    companion object {
        private const val TAG = "MainViewModel"
    }
}
