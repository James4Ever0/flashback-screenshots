package com.openmule.screenshotworker.service

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.PowerManager
import android.provider.Settings

/**
 * Helper to check and request battery optimization exemption.
 * Critical for keeping the foreground service alive on aggressive OEMs (Oppo, OnePlus, Xiaomi, Samsung).
 */
object BatteryOptimizationHelper {

    fun isIgnoringBatteryOptimizations(context: Context): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) return true
        val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        return pm.isIgnoringBatteryOptimizations(context.packageName)
    }

    /**
     * Opens the system battery optimization settings for this app.
     * Returns true if an intent was launched, false if not needed or unavailable.
     */
    fun requestIgnoreBatteryOptimizations(activity: Activity): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) return false
        if (isIgnoringBatteryOptimizations(activity)) return false

        val intent = Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS).apply {
            data = Uri.parse("package:${activity.packageName}")
        }
        if (intent.resolveActivity(activity.packageManager) != null) {
            activity.startActivity(intent)
            return true
        }
        return false
    }

    /**
     * Opens generic app battery settings as a fallback.
     */
    fun openBatterySettings(context: Context) {
        val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
            data = Uri.parse("package:${context.packageName}")
        }
        context.startActivity(intent)
    }
}
