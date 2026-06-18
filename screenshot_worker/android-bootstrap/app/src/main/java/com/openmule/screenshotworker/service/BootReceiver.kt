package com.openmule.screenshotworker.service

import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.widget.Toast
import androidx.core.app.NotificationCompat
import com.openmule.screenshotworker.MainActivity
import com.openmule.screenshotworker.ScreenshotWorkerApp
import com.openmule.screenshotworker.data.ConfigRepository
import com.openmule.screenshotworker.data.LogRepository
import java.io.PrintWriter
import java.io.StringWriter

/**
 * Handles two scenarios:
 * 1. BOOT_COMPLETED — device just booted. Cannot auto-start MediaProjection service
 *    (requires user interaction). Show a notification prompting user to open app.
 * 2. ACTION_RESTART_SERVICE — service was killed by OS. Same limitation: cannot
 *    auto-recover MediaProjection. Show recovery notification.
 */
class BootReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        try {
            val logger = LogRepository.getInstance()
            logger.i(TAG, "onReceive action=${intent.action}")

            when (intent.action) {
                Intent.ACTION_BOOT_COMPLETED -> {
                    try {
                        val config = ConfigRepository(context)
                        if (!config.autoStart) {
                            logger.i(TAG, "Boot completed, auto-start disabled")
                            return
                        }
                        logger.i(TAG, "Boot completed, auto-start enabled — showing notification")
                        showBootNotification(context)
                    } catch (e: Exception) {
                        logException(context, "BOOT_COMPLETED", e)
                    }
                }
                ScreenshotService.ACTION_RESTART_SERVICE -> {
                    try {
                        logger.w(TAG, "Service restart requested — showing recovery notification")
                        showRecoveryNotification(context)
                    } catch (e: Exception) {
                        logException(context, "RESTART_SERVICE", e)
                    }
                }
                else -> {
                    logger.d(TAG, "Unknown action: ${intent.action}")
                }
            }
        } catch (e: Exception) {
            // Last resort catch — BootReceiver must never crash
            try {
                val sw = StringWriter()
                e.printStackTrace(PrintWriter(sw))
                android.util.Log.e(TAG, "FATAL in onReceive: ${e.message}\n$sw")
            } catch (_: Exception) {
                // Nothing more we can do
            }
        }
    }

    private fun showBootNotification(context: Context) {
        try {
            val channelId = ScreenshotWorkerApp.CHANNEL_RECOVERY
            val nm = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

            val openAppIntent = Intent(context, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
            }
            val pendingIntent = PendingIntent.getActivity(
                context, BOOT_REQUEST_CODE, openAppIntent,
                PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
            )

            val notification = NotificationCompat.Builder(context, channelId)
                .setContentTitle("Screenshot Worker")
                .setContentText("Tap to start screen capture")
                .setSmallIcon(android.R.drawable.ic_menu_camera)
                .setContentIntent(pendingIntent)
                .setAutoCancel(true)
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .build()

            nm.notify(BOOT_NOTIFICATION_ID, notification)
        } catch (e: Exception) {
            logException(context, "showBootNotification", e)
            try {
                Toast.makeText(context, "Boot notification failed: ${e.message}", Toast.LENGTH_LONG).show()
            } catch (_: Exception) {}
        }
    }

    private fun showRecoveryNotification(context: Context) {
        try {
            val channelId = ScreenshotWorkerApp.CHANNEL_RECOVERY
            val nm = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

            val openAppIntent = Intent(context, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
            }
            val pendingIntent = PendingIntent.getActivity(
                context, RECOVERY_REQUEST_CODE, openAppIntent,
                PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
            )

            val notification = NotificationCompat.Builder(context, channelId)
                .setContentTitle("Screenshot Worker Stopped")
                .setContentText("Service was stopped. Tap to restart capture.")
                .setSmallIcon(android.R.drawable.ic_dialog_alert)
                .setContentIntent(pendingIntent)
                .setAutoCancel(true)
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .build()

            nm.notify(RECOVERY_NOTIFICATION_ID, notification)
        } catch (e: Exception) {
            logException(context, "showRecoveryNotification", e)
            try {
                Toast.makeText(context, "Recovery notification failed: ${e.message}", Toast.LENGTH_LONG).show()
            } catch (_: Exception) {}
        }
    }

    private fun logException(context: Context, where: String, e: Throwable) {
        try {
            val sw = StringWriter()
            e.printStackTrace(PrintWriter(sw))
            LogRepository.getInstance().e(TAG, "EXCEPTION in $where: ${e.message}\n$sw")
        } catch (_: Exception) {}
    }

    companion object {
        private const val TAG = "BootReceiver"
        private const val BOOT_NOTIFICATION_ID = 10
        private const val RECOVERY_NOTIFICATION_ID = 11
        private const val BOOT_REQUEST_CODE = 100
        private const val RECOVERY_REQUEST_CODE = 101
    }
}
