package com.openmule.screenshotworker

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.core.content.ContextCompat
import com.openmule.screenshotworker.data.LogRepository
import com.openmule.screenshotworker.service.ScreenshotService
import com.openmule.screenshotworker.ui.screens.HomeScreen
import com.openmule.screenshotworker.ui.theme.ScreenshotWorkerTheme
import com.openmule.screenshotworker.ui.viewmodel.MainViewModel

class MainActivity : ComponentActivity() {

    private val viewModel: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        checkNotificationPermission()
        handleAutoStart(intent)
        setContent {
            ScreenshotWorkerTheme {
                HomeScreen()
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleAutoStart(intent)
    }

    /** Called from recovery notification tap — triggers auto-start if service is not running. */
    private fun handleAutoStart(intent: Intent?) {
        val autoStart = intent?.getBooleanExtra(ScreenshotService.EXTRA_AUTO_START, false) ?: false
        if (autoStart && !viewModel.isRunning) {
            LogRepository.getInstance().i(TAG, "Auto-start requested from recovery notification")
            viewModel.pendingAutoStart = true
        }
    }

    override fun onResume() {
        super.onResume()
        // Re-check battery optimization status when user returns from settings
        viewModel.checkBatteryStatus(this)
        // Re-check notification permission in case user changed it in settings
        checkNotificationPermission()
    }

    /**
     * Android 13+ (API 33) requires runtime permission for posting notifications.
     * Without this, startForeground() will throw SecurityException and the service dies.
     */
    private fun checkNotificationPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            val granted = ContextCompat.checkSelfPermission(
                this, Manifest.permission.POST_NOTIFICATIONS
            ) == PackageManager.PERMISSION_GRANTED
            viewModel.onNotificationPermissionChecked(granted)
            if (!granted) {
                requestPermissions(
                    arrayOf(Manifest.permission.POST_NOTIFICATIONS),
                    REQ_POST_NOTIFICATIONS
                )
            }
        } else {
            viewModel.onNotificationPermissionChecked(true)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_POST_NOTIFICATIONS) {
            val granted = grantResults.isNotEmpty() &&
                    grantResults[0] == PackageManager.PERMISSION_GRANTED
            viewModel.onNotificationPermissionChecked(granted)
        }
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val REQ_POST_NOTIFICATIONS = 1001
    }
}
