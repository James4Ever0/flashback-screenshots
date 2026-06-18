// Package buffer implements a disk-based FIFO ring buffer for screenshot frames.
package buffer

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// Buffer is a directory-backed FIFO queue with a max file count.
type Buffer struct {
	dir      string
	maxFiles int
	mu       sync.Mutex
}

// New creates a new disk buffer at the given directory.
func New(dir string, maxFiles int) (*Buffer, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("mkdir %s: %w", dir, err)
	}
	// Test writability
	test := filepath.Join(dir, ".write_test")
	if err := os.WriteFile(test, []byte("test"), 0644); err != nil {
		return nil, fmt.Errorf("buffer dir not writable: %w", err)
	}
	os.Remove(test)

	return &Buffer{dir: dir, maxFiles: maxFiles}, nil
}

// Write saves a frame to disk using nanosecond timestamp as filename.
func (b *Buffer) Write(timestamp int64, data []byte) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	filename := fmt.Sprintf("%d.jpg", timestamp)
	path := filepath.Join(b.dir, filename)

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("write %s: %w", filename, err)
	}

	// Evict oldest if over limit
	files, _ := b.listFiles()
	for len(files) > b.maxFiles {
		oldest := files[0]
		os.Remove(filepath.Join(b.dir, oldest))
		files = files[1:]
	}

	return nil
}

// Oldest returns the filename and data of the oldest frame.
// Returns empty filename if buffer is empty.
func (b *Buffer) Oldest() (string, []byte, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	files, err := b.listFiles()
	if err != nil {
		return "", nil, err
	}
	if len(files) == 0 {
		return "", nil, nil
	}

	oldest := files[0]
	path := filepath.Join(b.dir, oldest)
	data, err := os.ReadFile(path)
	if err != nil {
		return "", nil, fmt.Errorf("read %s: %w", oldest, err)
	}
	return oldest, data, nil
}

// Remove deletes a frame from the buffer after successful upload.
func (b *Buffer) Remove(filename string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	path := filepath.Join(b.dir, filename)
	return os.Remove(path)
}

// Len returns the number of files in the buffer.
func (b *Buffer) Len() int {
	b.mu.Lock()
	defer b.mu.Unlock()

	files, _ := b.listFiles()
	return len(files)
}

// TotalSize returns the total bytes of all buffered files.
func (b *Buffer) TotalSize() int64 {
	b.mu.Lock()
	defer b.mu.Unlock()

	files, _ := b.listFiles()
	var total int64
	for _, f := range files {
		info, err := os.Stat(filepath.Join(b.dir, f))
		if err == nil {
			total += info.Size()
		}
	}
	return total
}

// TotalWritten returns the count of all files ever written (including evicted).
func (b *Buffer) TotalWritten() int64 {
	b.mu.Lock()
	defer b.mu.Unlock()

	files, _ := b.listFiles()
	return int64(len(files))
}

// listFiles returns sorted list of frame filenames (oldest first).
func (b *Buffer) listFiles() ([]string, error) {
	entries, err := os.ReadDir(b.dir)
	if err != nil {
		return nil, err
	}

	var files []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasSuffix(name, ".jpg") {
			files = append(files, name)
		}
	}

	// Sort by nanosecond timestamp (oldest first)
	sort.Slice(files, func(i, j int) bool {
		a, _ := strconv.ParseInt(strings.TrimSuffix(files[i], ".jpg"), 10, 64)
		b, _ := strconv.ParseInt(strings.TrimSuffix(files[j], ".jpg"), 10, 64)
		return a < b
	})

	return files, nil
}
