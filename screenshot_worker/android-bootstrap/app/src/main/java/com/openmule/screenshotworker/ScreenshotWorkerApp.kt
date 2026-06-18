package com.openmule.screenshotworker

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import com.openmule.screenshotworker.data.LogRepository
import com.openmule.screenshotworker.data.ServiceStatePersistence

/**
 * Application class for global initialization.
 * Creates notification channels early so they exist before any service starts.
 */
class ScreenshotWorkerApp : Application() {

    override fun onCreate() {
        super.onCreate()
        // Android Studio "Apply Changes" deploys preserve SharedPreferences while killing the
        // process. A stale `service_alive=true` can make the UI show "RUNNING" for ~10s.
        // Clear it now on every cold start before the UI reads it.
        if (ServiceStatePersistence.clearIfStale(this)) {
            LogRepository.getInstance().w("App", "Cleared stale service state from previous process")
        }
        createNotificationChannels()
        LogRepository.getInstance().i("App", "ScreenshotWorker initialized")
    }

    private fun createNotificationChannels() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return

        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

        // Channel for the foreground service (low importance, ongoing)
        val serviceChannel = NotificationChannel(
            CHANNEL_SERVICE,
            "Screenshot Worker Service",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Ongoing notification for screen capture service"
            setShowBadge(false)
        }

        // Channel for recovery/boot notifications (high importance, must be seen)
        val recoveryChannel = NotificationChannel(
            CHANNEL_RECOVERY,
            "Service Recovery",
            NotificationManager.IMPORTANCE_HIGH
        ).apply {
            description = "Notifications when the service needs to be restarted"
        }

        nm.createNotificationChannel(serviceChannel)
        nm.createNotificationChannel(recoveryChannel)
    }

    companion object {
        const val CHANNEL_SERVICE = "screenshot_worker_channel"
        const val CHANNEL_RECOVERY = "screenshot_worker_recovery"
    }
}
