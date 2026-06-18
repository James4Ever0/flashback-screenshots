package logging

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/rs/zerolog"
)

// Setup configures zerolog with the given level and optional file output.
func Setup(level string, logFile string) error {
	lvl, err := zerolog.ParseLevel(level)
	if err != nil {
		lvl = zerolog.InfoLevel
	}
	zerolog.SetGlobalLevel(lvl)
	zerolog.TimeFieldFormat = time.RFC3339

	var writers []zerolog.ConsoleWriter
	console := zerolog.ConsoleWriter{Out: os.Stdout, TimeFormat: time.RFC3339}
	writers = append(writers, console)

	if logFile != "" {
		dir := filepath.Dir(logFile)
		if dir != "" && dir != "." {
			if err := os.MkdirAll(dir, 0755); err != nil {
				return fmt.Errorf("failed to create log directory: %w", err)
			}
		}
		f, err := os.OpenFile(logFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			return fmt.Errorf("failed to open log file: %w", err)
		}
		writers = append(writers, zerolog.ConsoleWriter{Out: f, TimeFormat: time.RFC3339, NoColor: true})
	}

	if len(writers) == 1 {
		log := zerolog.New(console).With().Timestamp().Logger()
		zerolog.DefaultContextLogger = &log
	} else {
		multi := zerolog.MultiLevelWriter(writers[0], writers[1])
		log := zerolog.New(multi).With().Timestamp().Logger()
		zerolog.DefaultContextLogger = &log
	}

	return nil
}

// Logger returns the default logger.
func Logger() zerolog.Logger {
	return zerolog.Logger{}
}
