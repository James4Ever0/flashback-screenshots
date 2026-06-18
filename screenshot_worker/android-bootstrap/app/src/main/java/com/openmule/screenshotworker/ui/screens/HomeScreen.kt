package com.openmule.screenshotworker.ui.screens

import android.app.Activity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.openmule.screenshotworker.data.LogRepository
import com.openmule.screenshotworker.ui.viewmodel.MainViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(viewModel: MainViewModel = viewModel()) {
    val context = LocalContext.current
    val logs by viewModel.logs.collectAsState()
    val listState = rememberLazyListState()
    var selectedTab by remember { mutableIntStateOf(0) }
    val tabs = listOf("Log", "Config", "Diagnostics")

    // Auto-scroll to bottom when new logs arrive
    LaunchedEffect(logs.size) {
        if (logs.isNotEmpty()) {
            listState.animateScrollToItem(logs.size - 1)
        }
    }

    // MediaProjection launcher
    val projectionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        viewModel.onMediaProjectionResult(result.resultCode, result.data)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Screenshot Worker") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        ) {
            StatusBar(viewModel, projectionLauncher)
            NotificationPermissionBanner(viewModel)
            BatteryOptimizationBanner(viewModel)

            TabRow(selectedTabIndex = selectedTab) {
                tabs.forEachIndexed { index, title ->
                    Tab(
                        selected = selectedTab == index,
                        onClick = { selectedTab = index },
                        text = { Text(title) }
                    )
                }
            }

            when (selectedTab) {
                0 -> LogTab(logs, listState, viewModel)
                1 -> ConfigTab(viewModel)
                2 -> DiagnosticScreen(viewModel)
            }
        }
    }
}

@Composable
private fun StatusBar(
    viewModel: MainViewModel,
    projectionLauncher: androidx.activity.result.ActivityResultLauncher<android.content.Intent>
) {
    val context = LocalContext.current

    // Auto-start from recovery notification tap — triggers the same flow as pressing Start
    LaunchedEffect(viewModel.pendingAutoStart) {
        if (viewModel.pendingAutoStart) {
            viewModel.pendingAutoStart = false
            if (!viewModel.isRunning) {
                val intent = viewModel.requestMediaProjection(context as Activity)
                if (intent != null) {
                    projectionLauncher.launch(intent)
                }
            }
        }
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (viewModel.isRunning)
                MaterialTheme.colorScheme.primaryContainer
            else
                MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    text = if (viewModel.isRunning) "● Running" else "○ Stopped",
                    fontWeight = FontWeight.Bold,
                    color = if (viewModel.isRunning) Color(0xFF2E7D32) else Color.Gray
                )
                Text(
                    text = viewModel.clientName,
                    style = MaterialTheme.typography.bodySmall
                )
                if (!viewModel.notificationPermissionGranted) {
                    Text(
                        text = "Notification permission required",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error
                    )
                }
            }

            Button(
                onClick = {
                    if (viewModel.isRunning) {
                        viewModel.stopService()
                    } else {
                        val intent = viewModel.requestMediaProjection(context as Activity)
                        if (intent != null) {
                            projectionLauncher.launch(intent)
                        }
                    }
                },
                enabled = viewModel.notificationPermissionGranted || viewModel.isRunning,
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (viewModel.isRunning)
                        MaterialTheme.colorScheme.error
                    else
                        MaterialTheme.colorScheme.primary
                )
            ) {
                Text(if (viewModel.isRunning) "Stop" else "Start")
            }
        }
    }

    if (viewModel.showPermissionDenied) {
        AlertDialog(
            onDismissRequest = { viewModel.showPermissionDenied = false },
            title = { Text("Permission Required") },
            text = { Text("Screen capture permission was denied. The app needs this to capture screenshots.") },
            confirmButton = {
                TextButton(onClick = { viewModel.showPermissionDenied = false }) {
                    Text("OK")
                }
            }
        )
    }
}

@Composable
private fun NotificationPermissionBanner(viewModel: MainViewModel) {
    if (viewModel.notificationPermissionGranted) return

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 8.dp, vertical = 4.dp),
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFFFFEBEE)
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp)
        ) {
            Text(
                text = "Notification permission denied",
                fontWeight = FontWeight.Bold,
                color = Color(0xFFB71C1C)
            )
            Text(
                text = "Android 13+ requires notification permission for foreground services. " +
                        "The service cannot start without it. Please grant the permission in app settings.",
                style = MaterialTheme.typography.bodySmall,
                color = Color(0xFFB71C1C)
            )
        }
    }
}

@Composable
private fun BatteryOptimizationBanner(viewModel: MainViewModel) {
    if (!viewModel.batteryOptimized) return

    val context = LocalContext.current

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 8.dp, vertical = 4.dp),
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFFFFF3E0)
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp)
        ) {
            Text(
                text = "Battery optimization is on",
                fontWeight = FontWeight.Bold,
                color = Color(0xFFE65100)
            )
            Text(
                text = "The OS may stop this service to save power. Disable battery optimization for reliable capture.",
                style = MaterialTheme.typography.bodySmall,
                color = Color(0xFFE65100)
            )
            Spacer(modifier = Modifier.height(8.dp))
            Button(
                onClick = {
                    viewModel.requestBatteryOptimization(context as Activity)
                },
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFFE65100)
                )
            ) {
                Text("Disable Battery Optimization")
            }
        }
    }
}

@Composable
private fun LogTab(
    logs: List<LogRepository.LogEntry>,
    listState: androidx.compose.foundation.lazy.LazyListState,
    viewModel: MainViewModel
) {
    Column(modifier = Modifier.fillMaxSize()) {
        Text(
            text = "${logs.size} entries",
            style = MaterialTheme.typography.labelSmall,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp)
        )

        LazyColumn(
            state = listState,
            modifier = Modifier
                .weight(1f)
                .padding(horizontal = 8.dp),
        ) {
            itemsIndexed(logs) { _, entry ->
                LogLine(entry)
            }
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(8.dp),
            horizontalArrangement = Arrangement.End
        ) {
            TextButton(onClick = { viewModel.clearLogs() }) {
                Text("Clear")
            }
        }
    }
}

@Composable
private fun LogLine(entry: LogRepository.LogEntry) {
    val color = when (entry.level) {
        LogRepository.Level.ERROR -> Color(0xFFB71C1C)
        LogRepository.Level.WARN -> Color(0xFFF57F17)
        LogRepository.Level.INFO -> Color(0xFF1B5E20)
        LogRepository.Level.DEBUG -> Color.Gray
    }

    Text(
        text = "${entry.timestamp} [${entry.level.name.first()}] ${entry.tag}: ${entry.message}",
        fontFamily = FontFamily.Monospace,
        fontSize = 11.sp,
        color = color,
        lineHeight = 14.sp,
        modifier = Modifier.padding(vertical = 1.dp)
    )
}

@Composable
private fun ConfigTab(viewModel: MainViewModel) {
    val scrollState = rememberScrollState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text(
            text = "Server",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold
        )

        OutlinedTextField(
            value = viewModel.serverUrl,
            onValueChange = { viewModel.serverUrl = it },
            label = { Text("WebSocket URL") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true
        )

        OutlinedTextField(
            value = viewModel.clientName,
            onValueChange = { viewModel.clientName = it },
            label = { Text("Client Name") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true
        )

        HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

        Text(
            text = "Capture",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold
        )

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(
                value = viewModel.captureInterval,
                onValueChange = { viewModel.captureInterval = it.filter { c -> c.isDigit() } },
                label = { Text("Interval (sec)") },
                modifier = Modifier.weight(1f),
                singleLine = true
            )

            OutlinedTextField(
                value = viewModel.jpegQuality,
                onValueChange = { viewModel.jpegQuality = it.filter { c -> c.isDigit() } },
                label = { Text("Quality (1-100)") },
                modifier = Modifier.weight(1f),
                singleLine = true
            )
        }

        HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

        Text(
            text = "Buffer",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold
        )

        OutlinedTextField(
            value = viewModel.bufferMaxFiles,
            onValueChange = { viewModel.bufferMaxFiles = it.filter { c -> c.isDigit() } },
            label = { Text("Max Files") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true
        )

        HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

        Text(
            text = "Diagnostics",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold
        )

        OutlinedTextField(
            value = viewModel.diagnosticRefreshInterval,
            onValueChange = { viewModel.diagnosticRefreshInterval = it.filter { c -> c.isDigit() } },
            label = { Text("Auto-refresh interval (sec)") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true
        )

        HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Auto-start on boot")
            Switch(
                checked = viewModel.autoStart,
                onCheckedChange = { viewModel.autoStart = it }
            )
        }

        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
            modifier = Modifier.fillMaxWidth()
        ) {
            Column {
                Text("WiFi only upload")
                Text(
                    text = "Skip sending on cellular data",
                    style = MaterialTheme.typography.bodySmall,
                    color = Color.Gray
                )
            }
            Switch(
                checked = viewModel.wifiOnlyUpload,
                onCheckedChange = { viewModel.wifiOnlyUpload = it }
            )
        }

        Button(
            onClick = { viewModel.saveConfig() },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Save Configuration")
        }
    }
}
