package com.openmule.screenshotworker.service

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.PowerManager
import android.app.KeyguardManager
import com.openmule.screenshotworker.data.LogRepository

/**
 * Monitors screen lock / display state and provides a skip signal for capture.
 *
 * Listens to SCREEN_ON, SCREEN_OFF, and USER_PRESENT broadcasts so we log
 * transitions in both logcat (via LogRepository) and the app console.
 */
class ScreenStateHelper(private val context: Context) {

    private val logger = LogRepository.getInstance()
    private val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager
    private val keyguardManager = context.getSystemService(Context.KEYGUARD_SERVICE) as KeyguardManager

    private var lastLoggedState: String? = null

    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(ctx: Context?, intent: Intent?) {
            when (intent?.action) {
                Intent.ACTION_SCREEN_ON -> {
                    logger.i(TAG, "Screen ON — ${lockStatusString()}")
                }
                Intent.ACTION_SCREEN_OFF -> {
                    logger.i(TAG, "Screen OFF — capture paused")
                }
                Intent.ACTION_USER_PRESENT -> {
                    logger.i(TAG, "User unlocked device — capture resumed")
                }
            }
        }
    }

    /** Register broadcast receivers. Call from service onStartCommand. */
    fun register() {
        try {
            val filter = IntentFilter().apply {
                addAction(Intent.ACTION_SCREEN_ON)
                addAction(Intent.ACTION_SCREEN_OFF)
                addAction(Intent.ACTION_USER_PRESENT)
            }
            context.registerReceiver(receiver, filter)
            logger.i(TAG, "ScreenStateHelper registered — ${lockStatusString()}")
        } catch (e: Exception) {
            logger.e(TAG, "Failed to register screen receiver: ${e.message}")
        }
    }

    /** Unregister broadcast receivers. Call from service onDestroy. */
    fun unregister() {
        try {
            context.unregisterReceiver(receiver)
            logger.i(TAG, "ScreenStateHelper unregistered")
        } catch (e: Exception) {
            // Already unregistered or never registered
        }
    }

    /**
     * Returns true if the screen is on AND the device is not keyguard-locked.
     * If false, the caller should skip capture.
     */
    fun canCapture(): Boolean {
        val interactive = powerManager.isInteractive
        val locked = keyguardManager.isKeyguardLocked
        val canCapture = interactive && !locked

        val state = "interactive=$interactive, keyguardLocked=$locked, canCapture=$canCapture"
        // Log only when state changes to avoid spam
        if (state != lastLoggedState) {
            if (canCapture) {
                logger.i(TAG, "Screen ready — $state")
            } else {
                logger.w(TAG, "Screen blocked — $state")
            }
            lastLoggedState = state
        }
        return canCapture
    }

    private fun lockStatusString(): String {
        val interactive = powerManager.isInteractive
        val locked = keyguardManager.isKeyguardLocked
        return when {
            !interactive -> "display OFF"
            locked -> "display ON, locked"
            else -> "display ON, unlocked"
        }
    }

    companion object {
        private const val TAG = "ScreenStateHelper"
    }
}
