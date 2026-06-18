// Package capture handles screenshot capture and metadata extraction.
package capture

import (
	"bytes"
	"fmt"
	"image/jpeg"
	"time"

	"github.com/kbinani/screenshot"
	"github.com/rs/zerolog/log"
)

// Frame holds a captured screenshot with metadata.
type Frame struct {
	Timestamp      time.Time `json:"timestamp"`
	DisplayIndex   int       `json:"display_index"`
	Width          int       `json:"width"`
	Height         int       `json:"height"`
	ImageJPEG      []byte    `json:"-"`
	SessionLocked  bool      `json:"session_locked"`
}

// Capture takes a screenshot of the specified display and returns a Frame.
// displayIndex 0 is the primary display.
func Capture(displayIndex, quality int, locked bool) (*Frame, error) {
	n := screenshot.NumActiveDisplays()
	if n == 0 {
		return nil, fmt.Errorf("no active displays found")
	}
	if displayIndex >= n {
		return nil, fmt.Errorf("display index %d out of range (max %d)", displayIndex, n-1)
	}

	bounds := screenshot.GetDisplayBounds(displayIndex)
	img, err := screenshot.CaptureRect(bounds)
	if err != nil {
		return nil, fmt.Errorf("capture failed: %w", err)
	}

	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: quality}); err != nil {
		return nil, fmt.Errorf("jpeg encode failed: %w", err)
	}

	log.Debug().
		Int("display", displayIndex).
		Int("width", bounds.Dx()).
		Int("height", bounds.Dy()).
		Int("size_bytes", buf.Len()).
		Msg("captured frame")

	return &Frame{
		Timestamp:     time.Now().UTC(),
		DisplayIndex:  displayIndex,
		Width:         bounds.Dx(),
		Height:        bounds.Dy(),
		ImageJPEG:     buf.Bytes(),
		SessionLocked: locked,
	}, nil
}

// NumDisplays returns the number of active displays.
func NumDisplays() int {
	return screenshot.NumActiveDisplays()
}
