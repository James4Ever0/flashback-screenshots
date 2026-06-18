// Package buffer implements a disk-based ring buffer for frame storage.
package buffer

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/rs/zerolog/log"
)

// Buffer is a thread-safe disk-based ring buffer.
type Buffer struct {
	dir      string
	maxFiles int
	mu       sync.Mutex
}

// New creates a new disk buffer at the given directory.
func New(dir string, maxFiles int) (*Buffer, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create buffer dir: %w", err)
	}
	return &Buffer{dir: dir, maxFiles: maxFiles}, nil
}

// Write stores a frame in the buffer. If over maxFiles, oldest entries are removed.
func (b *Buffer) Write(timestamp time.Time, data []byte) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Trim if needed
	if err := b.trimUnsafe(); err != nil {
		return err
	}

	filename := fmt.Sprintf("%d.jpg", timestamp.UnixNano())
	path := filepath.Join(b.dir, filename)

	// Atomic write: temp then rename
	tmpPath := path + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write temp file: %w", err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("failed to rename temp file: %w", err)
	}

	log.Debug().Str("file", filename).Int("size", len(data)).Msg("buffered frame")
	return nil
}

// Oldest returns the oldest frame in the buffer (filename and data).
func (b *Buffer) Oldest() (string, []byte, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	files, err := b.listFilesUnsafe()
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
		return "", nil, fmt.Errorf("failed to read %s: %w", oldest, err)
	}

	return oldest, data, nil
}

// Remove deletes a frame from the buffer after successful send.
func (b *Buffer) Remove(filename string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	path := filepath.Join(b.dir, filename)
	if err := os.Remove(path); err != nil {
		return fmt.Errorf("failed to remove %s: %w", filename, err)
	}

	log.Debug().Str("file", filename).Msg("removed frame from buffer")
	return nil
}

// Len returns the current number of buffered frames.
func (b *Buffer) Len() (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	files, err := b.listFilesUnsafe()
	if err != nil {
		return 0, err
	}
	return len(files), nil
}

// Scan returns all buffered filenames sorted oldest first.
func (b *Buffer) Scan() ([]string, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	return b.listFilesUnsafe()
}

// listFilesUnsafe returns .jpg files sorted by nanosecond timestamp (oldest first).
func (b *Buffer) listFilesUnsafe() ([]string, error) {
	entries, err := os.ReadDir(b.dir)
	if err != nil {
		return nil, fmt.Errorf("failed to read buffer dir: %w", err)
	}

	var files []string
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".jpg") {
			continue
		}
		files = append(files, e.Name())
	}

	// Sort by nanosecond timestamp in filename
	sort.Slice(files, func(i, j int) bool {
		si, _ := strconv.ParseInt(strings.TrimSuffix(files[i], ".jpg"), 10, 64)
		sj, _ := strconv.ParseInt(strings.TrimSuffix(files[j], ".jpg"), 10, 64)
		return si < sj
	})

	return files, nil
}

// trimUnsafe removes oldest files if count exceeds maxFiles.
func (b *Buffer) trimUnsafe() error {
	files, err := b.listFilesUnsafe()
	if err != nil {
		return err
	}

	if len(files) < b.maxFiles {
		return nil
	}

	toDelete := len(files) - b.maxFiles + 1 // +1 to make room for the new file
	for i := 0; i < toDelete; i++ {
		path := filepath.Join(b.dir, files[i])
		if err := os.Remove(path); err != nil {
			log.Warn().Str("file", files[i]).Err(err).Msg("failed to trim old frame")
		} else {
			log.Debug().Str("file", files[i]).Msg("trimmed old frame")
		}
	}

	return nil
}
