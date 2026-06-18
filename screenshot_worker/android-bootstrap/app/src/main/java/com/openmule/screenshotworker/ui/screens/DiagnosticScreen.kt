package com.openmule.screenshotworker.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.openmule.screenshotworker.ui.viewmodel.MainViewModel
import kotlinx.coroutines.delay

@Composable
fun DiagnosticScreen(viewModel: MainViewModel = viewModel()) {
    val diagnostics by viewModel.diagnostics.collectAsState()
    val refreshInterval = viewModel.diagnosticRefreshInterval.toIntOrNull()?.coerceAtLeast(1) ?: 3

    // Auto-refresh trigger — recomposes every N seconds so uptime/ago counters update
    var tick by remember { mutableIntStateOf(0) }
    LaunchedEffect(tick, refreshInterval) {
        delay(refreshInterval * 1000L)
        tick++
    }

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 8.dp, vertical = 4.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Auto-refresh every ${refreshInterval}s",
                    style = MaterialTheme.typography.bodySmall,
                    color = Color.Gray
                )
                TextButton(onClick = { tick++ }) {
                    Text("Refresh")
                }
            }
        }

        // === SERVICE STATUS CARD ===
        item {
            DiagCard(title = "Service") {
                val statusText = if (diagnostics.serviceAlive) "● RUNNING" else "○ STOPPED"
                val statusColor = if (diagnostics.serviceAlive)
                    Color(0xFF2E7D32) else Color.Gray
                DiagRow("Status", statusText, valueColor = statusColor)

                if (diagnostics.serviceStartTime > 0) {
                    val fmt = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.getDefault())
                    DiagRow("Started at", fmt.format(java.util.Date(diagnostics.serviceStartTime)))
                } else {
                    DiagRow("Started at", "—")
                }

                if (diagnostics.serviceAlive && diagnostics.serviceStartTime > 0) {
                    val uptime = System.currentTimeMillis() - diagnostics.serviceStartTime
                    DiagRow("Uptime", viewModel.formatUptime(uptime))
                } else if (diagnostics.lastSessionUptimeMs > 0) {
                    DiagRow("Last session", viewModel.formatUptime(diagnostics.lastSessionUptimeMs))
                } else {
                    DiagRow("Uptime", "—")
                }

                DiagRow("Captures this session", diagnostics.totalCaptures.toString())
            }
        }

        // Capture Status
        item {
            DiagCard(title = "Capture") {
                DiagRow("This session", diagnostics.totalCaptures.toString())
                DiagRow("All time (disk)", diagnostics.bufferTotalWritten.toString())
                DiagRow("Last resolution", diagnostics.lastCaptureResolution.ifEmpty { "—" })
                DiagRow("Last size", if (diagnostics.lastCaptureSizeBytes > 0)
                    viewModel.formatBytes(diagnostics.lastCaptureSizeBytes.toLong()) else "—")
                if (diagnostics.lastCaptureTime > 0) {
                    val ago = (System.currentTimeMillis() - diagnostics.lastCaptureTime) / 1000
                    DiagRow("Last capture", "${ago}s ago")
                } else {
                    DiagRow("Last capture", "—")
                }
            }
        }

        // WebSocket Status
        item {
            DiagCard(title = "WebSocket") {
                DiagRow("Connected", if (diagnostics.websocketConnected) "● YES" else "○ NO",
                    valueColor = if (diagnostics.websocketConnected) Color(0xFF2E7D32) else Color(0xFFB71C1C))
                DiagRow("URL", diagnostics.websocketUrl.ifEmpty { "—" })
                DiagRow("Network", diagnostics.networkType.ifEmpty { "—" })
                if (diagnostics.uploadBlocked) {
                    DiagRow("Upload", "BLOCKED (WiFi only)", valueColor = Color(0xFFB71C1C))
                }
                DiagRow("Frames sent", diagnostics.totalFramesSent.toString())
                if (diagnostics.lastWebsocketSendTime > 0) {
                    val ago = (System.currentTimeMillis() - diagnostics.lastWebsocketSendTime) / 1000
                    DiagRow("Last sent", "${ago}s ago")
                } else {
                    DiagRow("Last sent", "—")
                }
            }
        }

        // Buffer Status
        item {
            DiagCard(title = "Disk Buffer") {
                DiagRow("Files in buffer", diagnostics.bufferFileCount.toString())
                DiagRow("Buffer size", viewModel.formatBytes(diagnostics.bufferTotalBytes))
                DiagRow("Total written (all time)", diagnostics.bufferTotalWritten.toString())
                DiagRow("Buffer dir", "app/files/buffer/")
            }
        }

        // Recent Files
        item {
            DiagCard(title = "Recent Captures") {
                Text(
                    text = "Filenames are nanosecond timestamps (e.g. 1234567890123.jpg)",
                    style = MaterialTheme.typography.bodySmall,
                    color = Color.Gray
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Use logcat filter 'DiskBuffer' to see each capture filename",
                    style = MaterialTheme.typography.bodySmall,
                    color = Color.Gray
                )
            }
        }

        // Errors
        if (diagnostics.lastError.isNotEmpty()) {
            item {
                DiagCard(title = "Last Error") {
                    DiagRow("Error", diagnostics.lastError, valueColor = Color(0xFFB71C1C))
                    if (diagnostics.lastErrorTime > 0) {
                        val ago = (System.currentTimeMillis() - diagnostics.lastErrorTime) / 1000
                        DiagRow("When", "${ago}s ago")
                    }
                }
            }
        }

        // Footer
        item {
            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}

@Composable
private fun DiagCard(
    title: String,
    cardColor: Color = MaterialTheme.colorScheme.surfaceVariant,
    content: @Composable ColumnScope.() -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = cardColor)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp)
        ) {
            Text(
                text = title,
                fontWeight = FontWeight.Bold,
                style = MaterialTheme.typography.titleSmall
            )
            Spacer(modifier = Modifier.height(8.dp))
            content()
        }
    }
}

@Composable
private fun DiagRow(
    label: String,
    value: String,
    valueColor: Color = MaterialTheme.colorScheme.onSurface
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = Color.Gray
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodySmall,
            fontFamily = FontFamily.Monospace,
            fontWeight = FontWeight.SemiBold,
            color = valueColor
        )
    }
}
