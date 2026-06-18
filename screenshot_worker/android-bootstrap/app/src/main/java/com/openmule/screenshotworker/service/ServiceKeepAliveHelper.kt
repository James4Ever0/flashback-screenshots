package com.openmule.screenshotworker.service

import android.app.AlarmManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.SystemClock
import com.openmule.screenshotworker.data.LogRepository
import java.io.PrintWriter
import java.io.StringWriter

/**
 * Uses AlarmManager to periodically verify the service is alive and restart it if needed.
 * setExactAndAllowWhileIdle bypasses Doze mode on API 23+.
 */
object ServiceKeepAliveHelper {

    private const val TAG = "ServiceKeepAliveHelper"
    private const val ALARM_REQUEST_CODE = 9876
    private const val INTERVAL_MS = 60_000L // 1 minute

    fun schedule(context: Context) {
        try {
            val alarmManager = context.getSystemService(Context.ALARM_SERVICE) as AlarmManager
            val intent = Intent(context, ScreenshotService::class.java)
            val pendingIntent = PendingIntent.getForegroundService(
                context,
                ALARM_REQUEST_CODE,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )

            val triggerAt = SystemClock.elapsedRealtime() + INTERVAL_MS

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                alarmManager.setExactAndAllowWhileIdle(
                    AlarmManager.ELAPSED_REALTIME_WAKEUP,
                    triggerAt,
                    pendingIntent
                )
            } else {
                alarmManager.setExact(
                    AlarmManager.ELAPSED_REALTIME_WAKEUP,
                    triggerAt,
                    pendingIntent
                )
            }
            LogRepository.getInstance().d(TAG, "Alarm scheduled for +${INTERVAL_MS}ms")
        } catch (e: Exception) {
            val sw = StringWriter()
            e.printStackTrace(PrintWriter(sw))
            LogRepository.getInstance().e(TAG, "schedule FAILED: ${e.message}\n$sw")
        }
    }

    fun cancel(context: Context) {
        try {
            val alarmManager = context.getSystemService(Context.ALARM_SERVICE) as AlarmManager
            val intent = Intent(context, ScreenshotService::class.java)
            val pendingIntent = PendingIntent.getForegroundService(
                context,
                ALARM_REQUEST_CODE,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )
            alarmManager.cancel(pendingIntent)
            pendingIntent.cancel()
            LogRepository.getInstance().d(TAG, "Alarm cancelled")
        } catch (e: Exception) {
            val sw = StringWriter()
            e.printStackTrace(PrintWriter(sw))
            LogRepository.getInstance().e(TAG, "cancel FAILED: ${e.message}\n$sw")
        }
    }
}