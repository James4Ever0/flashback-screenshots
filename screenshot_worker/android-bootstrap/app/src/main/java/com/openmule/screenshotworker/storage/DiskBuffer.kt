package com.openmule.screenshotworker.storage

import android.content.Context
import com.openmule.screenshotworker.data.LogRepository
import java.io.File
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.write

/**
 * Disk-based ring buffer for screenshot frames.
 * Stores JPEG files with nanosecond timestamps as filenames.
 */
class DiskBuffer(context: Context, private val maxFiles: Int) {

    private val dir = File(context.filesDir, "buffer").apply { mkdirs() }
    private val lock = ReentrantReadWriteLock()
    private val logger = LogRepository.getInstance()

    /** Total frames ever written (not just currently stored). */
    private val _totalWritten = AtomicLong(0)
    val totalWritten: Long get() = _totalWritten.get()

    /** Total bytes ever written. */
    private val _totalBytesWritten = AtomicLong(0)
    val totalBytesWritten: Long get() = _totalBytesWritten.get()

    /** Timestamp of the most recent write. */
    @Volatile
    var lastWriteTime: Long = 0
        private set

    init {
        logger.i(TAG, "Buffer dir: ${dir.absolutePath}, maxFiles: $maxFiles")
    }

    /** Write a frame to disk. Deletes oldest if over maxFiles. */
    fun write(timestampNs: Long, data: ByteArray) {
        lock.write {
            trimIfNeeded()
            val file = File(dir, "$timestampNs.jpg")
            file.writeBytes(data)
            _totalWritten.incrementAndGet()
            _totalBytesWritten.addAndGet(data.size.toLong())
            lastWriteTime = System.currentTimeMillis()
            logger.i(TAG, "Stored frame: ${file.name} (${data.size} bytes, " +
                    "resolution=${file.nameWithoutExtension}, totalWritten=$_totalWritten)")
        }
    }

    /** Read the oldest frame. Returns (filename, data) or null if empty. */
    fun oldest(): Pair<String, ByteArray>? {
        return lock.read {
            val files = sortedFiles()
            if (files.isEmpty()) return@read null
            val file = files.first()
            val data = file.readBytes()
            logger.d(TAG, "Reading oldest: ${file.name}")
            file.name to data
        }
    }

    /** Remove a frame after successful transmission. */
    fun remove(filename: String) {
        lock.write {
            File(dir, filename).delete()
            logger.d(TAG, "Removed frame: $filename")
        }
    }

    /** Current frame count. */
    fun count(): Int {
        return lock.read { sortedFiles().size }
    }

    /** List all frame filenames, newest first. */
    fun listAllFiles(): List<String> {
        return lock.read {
            sortedFiles().map { it.name }.reversed()
        }
    }

    /** Total size of all buffered files in bytes. */
    fun totalSizeBytes(): Long {
        return lock.read {
            sortedFiles().sumOf { it.length() }
        }
    }

    /** Check if buffer is at capacity. */
    fun isFull(): Boolean {
        return lock.read { sortedFiles().size >= maxFiles }
    }

    /** Delete all frames. */
    fun clear() {
        lock.write {
            dir.listFiles()?.forEach { it.delete() }
            logger.i(TAG, "Buffer cleared")
        }
    }

    private fun sortedFiles(): List<File> {
        return dir.listFiles { f -> f.extension == "jpg" }
            ?.sortedBy { it.nameWithoutExtension.toLongOrNull() ?: 0L }
            ?: emptyList()
    }

    private fun trimIfNeeded() {
        val files = sortedFiles()
        if (files.size >= maxFiles) {
            val toDelete = files.size - maxFiles + 1
            files.take(toDelete).forEach {
                it.delete()
                logger.w(TAG, "Buffer full, trimmed: ${it.name}")
            }
        }
    }

    companion object {
        private const val TAG = "DiskBuffer"
    }
}
