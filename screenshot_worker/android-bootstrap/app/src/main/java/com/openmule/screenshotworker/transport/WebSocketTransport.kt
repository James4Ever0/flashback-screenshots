package com.openmule.screenshotworker.transport

import com.openmule.screenshotworker.data.DiagnosticRepository
import com.openmule.screenshotworker.data.LogRepository
import com.openmule.screenshotworker.storage.DiskBuffer
import kotlinx.coroutines.*
import okhttp3.*
import okio.ByteString
import org.json.JSONObject
import java.io.PrintWriter
import java.io.StringWriter
import java.util.concurrent.TimeUnit

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities

/**
 * WebSocket client that connects to the server and drains the disk buffer.
 * Runs fully independent from screenshot capture — capture never waits for websocket.
 * Auto-reconnects with exponential backoff. Connection/read/write timeouts applied.
 *
 * @param context Android context for checking network connectivity
 * @param wifiOnlyUpload When true, frames are only sent while on WiFi
 */
class WebSocketTransport(
    private val context: Context,
    private val serverUrl: String,
    private val clientName: String,
    private val buffer: DiskBuffer,
    private val wifiOnlyUpload: Boolean = true
) {
    private val logger = LogRepository.getInstance()
    private val diag = DiagnosticRepository.getInstance()

    private val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .writeTimeout(10, TimeUnit.SECONDS)
        .pingInterval(30, TimeUnit.SECONDS)
        .build()

    private var webSocket: WebSocket? = null
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var isRunning = false

    @Volatile
    private var _connected = false
    val connected: Boolean get() = _connected

    fun start() {
        if (isRunning) return
        isRunning = true
        logger.i(TAG, "WebSocket transport starting — url=$serverUrl")
        diag.update { copy(websocketUrl = serverUrl) }
        scope.launch {
            try {
                connectLoop()
            } catch (e: Exception) {
                logException("connectLoop", e)
            }
        }
        scope.launch {
            try {
                drainLoop()
            } catch (e: Exception) {
                logException("drainLoop", e)
            }
        }
    }

    fun stop() {
        try {
            logger.i(TAG, "WebSocket transport stopping")
            isRunning = false
            _connected = false
            diag.update { copy(websocketConnected = false) }
            webSocket?.close(1000, "Client stopping")
            scope.cancel()
        } catch (e: Exception) {
            logException("stop", e)
        }
    }

    /** Manages the websocket connection lifecycle. */
    private suspend fun connectLoop() {
        var backoffMs = 1000L
        val maxBackoff = 60000L

        while (isRunning && scope.isActive) {
            try {
                val request = Request.Builder()
                    .url("$serverUrl?name=$clientName")
                    .build()

                val latch = CompletableDeferred<Boolean>()
                webSocket = client.newWebSocket(request, object : WebSocketListener() {
                    override fun onOpen(ws: WebSocket, response: Response) {
                        try {
                            logger.i(TAG, "Connected to $serverUrl")
                            _connected = true
                            diag.update { copy(websocketConnected = true) }
                            ws.send(JSONObject().apply {
                                put("type", "hello")
                                put("name", clientName)
                                put("version", "0.1.0")
                            }.toString())
                            latch.complete(true)
                        } catch (e: Exception) {
                            logException("onOpen", e)
                            latch.complete(false)
                        }
                    }

                    override fun onMessage(ws: WebSocket, text: String) {
                        try {
                            handleMessage(text)
                        } catch (e: Exception) {
                            logException("onMessage text", e)
                        }
                    }

                    override fun onMessage(ws: WebSocket, bytes: ByteString) {
                        try {
                            logger.d(TAG, "Binary message received: ${bytes.size} bytes")
                        } catch (e: Exception) {
                            logException("onMessage bytes", e)
                        }
                    }

                    override fun onClosing(ws: WebSocket, code: Int, reason: String) {
                        try {
                            logger.w(TAG, "Server closing: $reason")
                            ws.close(code, reason)
                        } catch (e: Exception) {
                            logException("onClosing", e)
                        }
                    }

                    override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                        try {
                            logger.w(TAG, "Connection closed: $reason")
                            _connected = false
                            diag.update { copy(websocketConnected = false) }
                            latch.complete(false)
                        } catch (e: Exception) {
                            logException("onClosed", e)
                        }
                    }

                    override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                        try {
                            logException("onFailure", t)
                            _connected = false
                            diag.update {
                                copy(
                                    websocketConnected = false,
                                    lastError = "WS: ${t.message}",
                                    lastErrorTime = System.currentTimeMillis()
                                )
                            }
                            latch.complete(false)
                        } catch (e: Exception) {
                            logException("onFailure handler", e)
                        }
                    }
                })

                val connected = latch.await()
                if (connected) {
                    backoffMs = 1000L // Reset backoff on success
                    // Wait here until connection closes or transport stops
                    while (_connected && isRunning && scope.isActive) {
                        delay(1000)
                    }
                }
            } catch (e: Exception) {
                logException("connectLoop iteration", e)
                _connected = false
                diag.update {
                    copy(
                        websocketConnected = false,
                        lastError = "WS connect: ${e.message}",
                        lastErrorTime = System.currentTimeMillis()
                    )
                }
            }

            if (!isRunning) break

            logger.i(TAG, "Reconnecting in ${backoffMs}ms...")
            delay(backoffMs)
            backoffMs = (backoffMs * 2).coerceAtMost(maxBackoff)
        }
    }

    /** Check current active network type. Returns "WiFi", "Cellular", "None", or "Unknown". */
    private fun getNetworkType(): String {
        return try {
            val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
            val network = cm.activeNetwork ?: return "None"
            val caps = cm.getNetworkCapabilities(network) ?: return "None"
            when {
                caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> "WiFi"
                caps.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> "Cellular"
                caps.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET) -> "Ethernet"
                else -> "Unknown"
            }
        } catch (e: Exception) {
            logException("getNetworkType", e)
            "Unknown"
        }
    }

    /** True if currently connected to a network that allows upload per [wifiOnlyUpload] setting. */
    private fun canUpload(): Boolean {
        if (!wifiOnlyUpload) return true
        val type = getNetworkType()
        return type == "WiFi" || type == "Ethernet"
    }

    /** Independent loop that drains the disk buffer. Never blocks capture. */
    private suspend fun drainLoop() {
        while (isRunning && scope.isActive) {
            val netType = getNetworkType()
            val blocked = wifiOnlyUpload && !(netType == "WiFi" || netType == "Ethernet")
            diag.update {
                copy(networkType = netType, uploadBlocked = blocked)
            }

            if (!_connected) {
                delay(500)
                continue
            }

            if (blocked) {
                logger.i(TAG, "Upload blocked: on $netType, waiting for WiFi (wifiOnlyUpload=$wifiOnlyUpload)")
                delay(5_000)
                continue
            }

            val frame = try {
                buffer.oldest()
            } catch (e: Exception) {
                logException("buffer.oldest()", e)
                null
            } ?: run {
                delay(500)
                continue
            }

            val (filename, data) = frame

            val sent = try {
                val json = JSONObject().apply {
                    put("type", "frame")
                    put("filename", filename)
                    put("timestamp", System.currentTimeMillis())
                }
                val base64Data = java.util.Base64.getEncoder().encodeToString(data)
                json.put("data", base64Data)
                val payload = json.toString()
                val ok = webSocket?.send(payload) ?: false
                if (ok) {
                    logger.i(TAG, "Sent frame: $filename (${data.size} bytes, payload=${payload.length} chars)")
                }
                ok
            } catch (e: Exception) {
                logException("send frame $filename", e)
                false
            }

            if (sent) {
                try {
                    buffer.remove(filename)
                    val now = System.currentTimeMillis()
                    diag.update {
                        copy(
                            lastWebsocketSendTime = now,
                            totalFramesSent = totalFramesSent + 1
                        )
                    }
                } catch (e: Exception) {
                    logException("buffer.remove after send", e)
                }
            } else {
                logger.w(TAG, "Send failed, keeping frame: $filename")
                delay(1000)
            }
        }
    }

    private fun handleMessage(text: String) {
        try {
            val json = JSONObject(text)
            when (json.optString("type")) {
                "ping" -> webSocket?.send(JSONObject().put("type", "pong").toString())
                "ack" -> logger.i(TAG, "Server ACK: ${json.optString("filename")}")
                else -> logger.d(TAG, "Server msg: $text")
            }
        } catch (e: Exception) {
            logger.d(TAG, "Raw server msg: $text")
        }
    }

    private fun logException(context: String, e: Throwable) {
        val sw = StringWriter()
        e.printStackTrace(PrintWriter(sw))
        logger.e(TAG, "EXCEPTION in $context: ${e.message}\n$sw")
    }

    companion object {
        private const val TAG = "WebSocketTransport"
    }
}
