// Package screencap wraps Android's /system/bin/screencap binary.
package screencap

import (
	"bytes"
	"fmt"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"

	"android-go-bootstrap-su/internal/root"
)

// Capture takes a screenshot of the given display and returns JPEG bytes.
// It uses /system/bin/screencap via su, matching the Python reference.
func Capture(displayID int, quality int) (*Result, error) {
	if !root.CheckRoot() {
		return nil, fmt.Errorf("root not available")
	}

	tmpPath := fmt.Sprintf("/data/local/tmp/go_screenshot_%d_%d.png", os.Getpid(), timeNanos())
	defer root.RunSu("rm -f " + tmpPath)

	// Build screencap command. -d flag requires Android 11+ (API 30).
	screencapCmd := fmt.Sprintf("/system/bin/screencap -p -d %d %s", displayID, tmpPath)
	out, err := root.RunSu(screencapCmd)
	if err != nil {
		// Fallback: try without -d for older Android
		screencapCmd = fmt.Sprintf("/system/bin/screencap -p %s", tmpPath)
		out, err = root.RunSu(screencapCmd)
		if err != nil {
			return nil, fmt.Errorf("screencap failed: %w (output: %s)", err, string(out))
		}
	}

	// Read the captured PNG
	pngData, err := os.ReadFile(tmpPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read screenshot: %w", err)
	}

	// Decode PNG
	img, err := png.Decode(bytes.NewReader(pngData))
	if err != nil {
		return nil, fmt.Errorf("failed to decode PNG: %w", err)
	}

	// Encode to JPEG with quality
	bounds := img.Bounds()
	var jpegBuf bytes.Buffer
	if err := jpeg.Encode(&jpegBuf, img, &jpeg.Options{Quality: quality}); err != nil {
		return nil, fmt.Errorf("jpeg encode failed: %w", err)
	}

	return &Result{
		Width:       bounds.Dx(),
		Height:      bounds.Dy(),
		JPEG:        jpegBuf.Bytes(),
		DisplayID:   displayID,
	}, nil
}

// Result holds a captured screenshot.
type Result struct {
	Width     int
	Height    int
	JPEG      []byte
	DisplayID int
}

// SaveTo writes the screenshot to the given path, fixing permissions.
func SaveTo(path string, displayID int) error {
	if !root.CheckRoot() {
		return fmt.Errorf("root not available")
	}

	tmpPath := fmt.Sprintf("/data/local/tmp/go_screenshot_save_%d_%d.png", os.Getpid(), timeNanos())
	defer root.RunSu("rm -f " + tmpPath)

	screencapCmd := fmt.Sprintf("/system/bin/screencap -p -d %d %s", displayID, tmpPath)
	out, err := root.RunSu(screencapCmd)
	if err != nil {
		screencapCmd = fmt.Sprintf("/system/bin/screencap -p %s", tmpPath)
		out, err = root.RunSu(screencapCmd)
		if err != nil {
			return fmt.Errorf("screencap failed: %w (output: %s)", err, string(out))
		}
	}

	// Ensure target directory exists
	dir := filepath.Dir(path)
	if dir != "" && dir != "." {
		root.RunSu("mkdir -p " + dir)
	}

	// Copy to destination
	out, err = root.RunSu(fmt.Sprintf("cp %s %s", tmpPath, path))
	if err != nil {
		return fmt.Errorf("copy failed: %w (output: %s)", err, string(out))
	}

	// Fix permissions to current user
	whoamiOut, _ := root.RunSu("whoami")
	user := string(whoamiOut)
	if user == "" {
		user = "shell"
	}
	root.RunSu(fmt.Sprintf("chown %s:%s %s", user, user, path))

	return nil
}

func timeNanos() int64 {
	return int64(os.Getpid()) ^ (int64(os.Getuid()) << 32)
}
